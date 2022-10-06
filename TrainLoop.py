import logging
import pickle
import random
import shutil
from torch.multiprocessing import Process, Lock, Queue
from time import sleep
import torch, os
from Simulation import Simulation
from agents.RLAgents import *
import argparse
import tqdm
import numpy as np

ctx = torch.multiprocessing.get_context('spawn')
global_main_queue = ctx.Queue(maxsize=10000)
global_chaodi_queue = ctx.Queue(maxsize=10000)
global_declare_queue = ctx.Queue(maxsize=10000)
global_kitty_queue = ctx.Queue(maxsize=10000)
actor_processes = []

# Parallelized data sampling
def sampler(idx: int, main_agent, declare_agent, kitty_agent, chaodi_agent, eval_main, eval_declare, eval_kitty, eval_chaodi, discount, decay_factor, global_main_queue, global_chaodi_queue, global_declare_queue, global_kitty_queue, combos):
    train_sim = Simulation(
        main_agent=main_agent,
        declare_agent=declare_agent,
        kitty_agent=kitty_agent,
        chaodi_agent=chaodi_agent,
        enable_combos=combos,
        epsilon=0.2
    )
    while True:
        local_main, local_declare, local_kitty, local_chaodi = [], [], [], []
        for i in range(10):
            with torch.no_grad():
                while train_sim.step()[0]: pass
                local_main.extend(train_sim.main_history[:])
                local_declare.extend(train_sim.declaration_history[:])
                local_chaodi.extend(train_sim.chaodi_history[:])
                local_kitty.extend(train_sim.kitty_history[:])
            
            train_sim.reset()
            train_sim.epsilon = max(0.01, train_sim.epsilon / decay_factor)
        global_main_queue.put(local_main)
        global_declare_queue.put(local_declare)
        global_chaodi_queue.put(local_chaodi)
        global_kitty_queue.put(local_kitty)

def evaluator(idx: int, main_agent, declare_agent, kitty_agent, chaodi_agent, eval_main, eval_declare, eval_kitty, eval_chaodi, combos, eval_size, eval_results_queue: Queue):
    eval_sim = Simulation(
        main_agent=main_agent,
        declare_agent=declare_agent,
        kitty_agent=kitty_agent,
        chaodi_agent=chaodi_agent,
        enable_combos=combos,
        eval_main=eval_main,
        eval_declare=eval_declare,
        eval_kitty=eval_kitty,
        eval_chaodi=eval_chaodi,
        eval=True
    )
    for _ in range(eval_size):
        with torch.no_grad():
            while eval_sim.step()[0]: pass
        opponent_index = int(eval_sim.game_engine.dealer_position in ['N', 'S'])
        opponents_won = eval_sim.game_engine.opponent_points >= 80
        eval_results_queue.put((int(opponents_won) if opponent_index == 1 else (1 - opponents_won), opponent_index, eval_sim.game_engine.opponent_points))
        eval_sim.reset()
    exit(0)


def train(games: int, model_folder: str, eval_only: bool, eval_size: int, compare: str = None, discount=0.99, decay_factor=1.2, combos=False, verbose=False, random_seed=1):
    os.makedirs(model_folder, exist_ok=True)
    torch.manual_seed(0)
    random.seed(random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    declare_model = DeclarationModel().to(device)
    if os.path.exists(f'{model_folder}/declare.pt'):
        declare_agent = declare_model.load_state_dict(torch.load(f'{model_folder}/declare.pt', map_location=device), strict=False)
        print("Using loaded model for declaration")
    declare_agent = DeclareAgent('declare', declare_model)
    
    kitty_model = KittyModel().to(device)
    if os.path.exists(f'{model_folder}/kitty.pt'):
        kitty_model.load_state_dict(torch.load(f'{model_folder}/kitty.pt', map_location=device), strict=False)
        print("Using loaded model for kitty")
    kitty_agent = KittyAgent('kitty', kitty_model)
    
    chaodi_model = ChaodiModel().to(device)
    if os.path.exists(f'{model_folder}/chaodi.pt'):
        chaodi_model.load_state_dict(torch.load(f'{model_folder}/chaodi.pt', map_location=device), strict=False)
        print("Using loaded model for chaodi")
    chaodi_agent = ChaodiAgent('chaodi', chaodi_model)

    main_model = MainModel().to(device)
    if os.path.exists(f'{model_folder}/main.pt'):
        main_model.load_state_dict(torch.load(f'{model_folder}/main.pt', map_location=device), strict=False)
        print("Using loaded model for main game")
    main_agent = MainAgent('main', main_model)
    
    declare_model.share_memory()
    kitty_model.share_memory()
    chaodi_model.share_memory()
    main_model.share_memory()

    eval_main, eval_kitty, eval_chaodi, eval_declare = None, None, None, None
    if compare:
        eval_main_model = MainModel().to(device)
        eval_main_model.load_state_dict(torch.load(f'{compare}/main.pt', map_location=device), strict=False)
        print(f"Using main model in {compare} for comparison")
        eval_main = MainAgent("main", eval_main_model)

        eval_kitty_model = KittyModel().to(device)
        eval_kitty_model.load_state_dict(torch.load(f'{compare}/kitty.pt', map_location=device), strict=False)
        print(f"Using kitty model in {compare} for comparison")
        eval_kitty = KittyAgent("kitty", eval_kitty_model)

        eval_declare_model = DeclarationModel().to(device)
        eval_declare_model.load_state_dict(torch.load(f'{compare}/declare.pt', map_location=device), strict=False)
        print(f"Using declare model in {compare} for comparison")
        eval_declare = DeclareAgent("declare", eval_declare_model)

        eval_chaodi_model = ChaodiModel().to(device)
        eval_chaodi_model.load_state_dict(torch.load(f'{compare}/chaodi.pt', map_location=device), strict=False)
        print(f"Using chaodi model in {compare} for comparison")
        eval_chaodi = ChaodiAgent('chaodi', eval_chaodi_model)

        eval_main_model.share_memory()
        eval_kitty_model.share_memory()
        eval_declare_model.share_memory()
        eval_chaodi_model.share_memory()

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.WARN)
    
    stats = []
    iterations = 0

    try:
        with open(f'{model_folder}/state.pkl', mode='rb') as f:
            state = pickle.load(f)
            iterations = state['iterations']
            print(f"Using checkpoint at iteration {iterations}")
        with open(f'{model_folder}/stats.pkl', mode='rb') as f:
            stats = pickle.load(f)
            print(f"Continuing with saved stats")
    except:
        print("Starting new training session")
        shutil.copyfile('networks/Models.py', f'{model_folder}/Models.py')
    
    if not eval_only:
        for i in range(4):
            actor = ctx.Process(target=sampler, args=(i, main_agent, declare_agent, kitty_agent, chaodi_agent, eval_main, eval_declare, eval_kitty, eval_chaodi, discount, decay_factor ** (1 / games), global_main_queue, global_chaodi_queue, global_declare_queue, global_kitty_queue, combos))
            actor.start()
            actor_processes.append(actor)
            
            print(f"Spawned process {i}")
    
  
    while True:
        if not eval_only:
            print(f"Training iterations {iterations * games}-{(iterations+1) * games}...")
            for _ in tqdm.tqdm(range(0, games, 10)):
                # print('wait', global_main_queue.qsize(), global_declare_queue.qsize(), global_kitty_queue.qsize(), global_chaodi_queue.qsize())
                declare_batch = global_declare_queue.get()
                kitty_batch = global_kitty_queue.get()
                main_batch = global_main_queue.get()
                chaodi_batch = global_chaodi_queue.get()
                declare_agent.learn_from_samples(declare_batch)
                kitty_agent.learn_from_samples(kitty_batch)
                chaodi_agent.learn_from_samples(chaodi_batch)
                main_agent.learn_from_samples(main_batch)
            for agent in (declare_agent, kitty_agent, chaodi_agent, main_agent):
                torch.save(agent.model.state_dict(), f'{model_folder}/{agent.name}.pt')
            print('main loss:', np.mean(main_agent.train_loss_history), 'declare loss:', np.mean(declare_agent.train_loss_history), 'kitty loss:', np.mean(kitty_agent.train_loss_history), 'chaodi loss:', np.mean(chaodi_agent.train_loss_history))
            main_agent.train_loss_history.clear()
            declare_agent.train_loss_history.clear()
            kitty_agent.train_loss_history.clear()
            chaodi_agent.train_loss_history.clear()

        # for i in tqdm.tqdm(range(games)):
        #     with torch.no_grad():
        #         while train_sim.step()[0]: pass
        #     if not eval_only:
        #         if (i+1) % 5 == 0:
        #             train_sim.backprop()
        #             for agent in (declare_agent, kitty_agent, chaodi_agent, main_agent):
        #                 torch.save(agent.model.state_dict(), f'{model_folder}/{agent.name}.pt')
        #     train_sim.reset()
        
        print("Evaluating model performance...")
        # eval_sim = Simulation(
        #     main_agent=main_agent,
        #     declare_agent=declare_agent,
        #     kitty_agent=kitty_agent,
        #     chaodi_agent=chaodi_agent,
        #     enable_combos=combos,
        #     eval=True
        # )
        # for _ in tqdm.tqdm(range(eval_size)):
        #     with torch.no_grad():
        #         while eval_sim.step()[0]: pass
        #     eval_sim.reset()
        eval_size = eval_size // 4 * 4 # Must be multiple of 4
        win_counts = [0, 0] # Defenders, opponents
        opposition_points = [[], []]
        eval_queue = ctx.Queue()
        eval_actors = []
        for i in range(4):
            actor = ctx.Process(target=evaluator, args=(i, main_agent, declare_agent, kitty_agent, chaodi_agent, eval_main, eval_declare, eval_kitty, eval_chaodi, combos, eval_size // 4, eval_queue))
            actor.start()
            eval_actors.append(actor)
        

        with tqdm.tqdm(total=eval_size) as progress_bar:
            for i in range(eval_size):
                win_index, opponent_index, points = eval_queue.get()
                win_counts[win_index] += 1
                opposition_points[opponent_index].append(points)
                progress_bar.update(1)
                
        
        for a in eval_actors:
            a.join()

        print('Win counts:', win_counts)
        print("Average opposition points:", np.mean(opposition_points[0]), np.mean(opposition_points[1]))
        
        iterations += 1

        if not eval_only:
            stats.append({
                "iterations": iterations * games,
                "win_counts": win_counts[0] / sum(win_counts),
                "avg_points": [np.mean(opposition_points[0]), np.mean(opposition_points[1])]
            })
            with open(f'{model_folder}/stats.pkl', mode='w+b') as f:
                pickle.dump(stats, f)
            with open(f'{model_folder}/state.pkl', mode='w+b') as f:
                pickle.dump({
                    'iterations': iterations,
                    'main_optim_state': main_agent.optimizer.state_dict(),
                    'declare_optim_state': declare_agent.optimizer.state_dict(),
                    'kitty_optim_state': kitty_agent.optimizer.state_dict(),
                    'chaodi_optim_state': chaodi_agent.optimizer.state_dict()
                }, f)
        else:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train loop')
    parser.add_argument('--games', type=int, default=500)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-size', type=int, default=300)
    parser.add_argument('--model-folder', type=str, default='pretrained')
    parser.add_argument('--compare', type=str, default='')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--decay-factor', type=float, default=1.2)
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--enable-combos', action='store_true')
    # parser.add_argument('--epsilon', type=float, default=0.02)
    args = parser.parse_args()
    train(args.games, args.model_folder, args.eval_only, args.eval_size, args.compare, args.discount, args.decay_factor, args.enable_combos, args.verbose, args.random_seed)