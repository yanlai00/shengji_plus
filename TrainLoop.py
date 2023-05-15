import importlib
import logging
import pickle
import random
import shutil
from torch.multiprocessing import Process, Lock, Queue
import torch, os, sys
from Simulation import Simulation
from agents.Agent import SJAgent
from agents.RandomAgent import RandomAgent
from agents.StrategicAgent import StrategicAgent
from agents.InteractiveAgent import InteractiveAgent
from agents.DMCAgent import DMCAgent
from agents.DQNAgent import DQNAgent
from env.utils import Stage
import argparse
import tqdm
import numpy as np
from torch import nn

ctx = torch.multiprocessing.get_context('spawn')
global_main_queue = ctx.Queue(maxsize=25)
actor_processes = []

# Parallelized data sampling
def sampler(idx: int, player: SJAgent, discount, decay_factor, global_main_queue, epsilon=0.02, reuse_times=0, oracle_duration=0, game_count=0, log_file=''):
    logging.getLogger().setLevel(logging.ERROR)
    # logging.basicConfig(format="%(process)d %(message)s", filename=log_file, encoding='utf-8', level=logging.DEBUG)
    train_sim = Simulation(
        player1=player,
        player2=None,
        discount=discount,
        epsilon=0.02,
        oracle_duration=oracle_duration,
        game_count=game_count,
    )
    while True:
        local_main = []
        same_deck_count = 0
        for i in range(10):
            with torch.no_grad():
                while train_sim.step()[0]: pass
                local_main.extend(train_sim.main_history)
            
            # Get new deck every `reuse_times` times
            if same_deck_count < reuse_times:
                same_deck_count += 1
                train_sim.reset(reuse_old_deck=True)
            else:
                same_deck_count = 0
                train_sim.reset(reuse_old_deck=False)
            # with open(log_file, 'w') as f:
            #     f.write('')
        train_sim.epsilon = max(epsilon, train_sim.epsilon / decay_factor)
        global_main_queue.put(local_main)

def evaluator(idx: int, player1: SJAgent, player2: SJAgent, eval_size: int, eval_results_queue: Queue, verbose=False, learn_from_eval=False, log_file=''):
    logging.getLogger().setLevel(logging.ERROR)
    # if not verbose:
    #     logging.basicConfig(format="%(process)d %(message)s", filename=log_file, encoding='utf-8', level=logging.DEBUG)
    random.seed(idx)
    eval_sim = Simulation(
        player1=player1,
        player2=player2,
        eval=True,
        learn_from_eval=learn_from_eval
    )
    iterations = 0
    while True:
        with torch.no_grad():
            while eval_sim.step()[0]: pass
        opponent_index = int(eval_sim.game_engine.dealer_position in ['N', 'S'])
        opponents_won = eval_sim.game_engine.opponent_points >= 80
        win_index = int(opponents_won) if opponent_index == 1 else (1 - opponents_won)
        eval_results_queue.put((
            win_index,
            opponent_index,
            eval_sim.game_engine.opponent_points,
            abs(eval_sim.game_engine.final_defender_reward)
        ))
        eval_sim.reset()
        iterations += 1
        
        # if iterations > eval_size:
        #     exit(0)
    


def train(agent_type: str, games: int, model_folder: str, eval_only: bool, eval_size: int, compare: str = None, discount=0.99, decay_factor=1.2, verbose=False, random_seed=1, single_process=False, epsilon=0.01, tau=0.995, eval_agent_type='random', learn_from_eval=False, reuse_times=0, oracle_duration=0, max_games=500000, dynamic_encoding=True, actor_process_count=6, eval_process_count=7):
    os.makedirs(model_folder, exist_ok=True)
    torch.manual_seed(0)
    random.seed(random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    oracle_duration_input = oracle_duration

    try:
        train_models = importlib.import_module(f"{model_folder}.Models".replace('/', '.'))
    except:
        train_models = importlib.import_module("networks.Models")

    agent: SJAgent
    iterations = 0
    stats = []

    if agent_type.startswith('dmc'):
        agent = DMCAgent(model_folder, use_oracle=oracle_duration_input > 0, dynamic_encoding=dynamic_encoding, sac=agent_type.endswith('sac'))
        print(f"Using DMC model {'with' if dynamic_encoding else 'without'} dynamic encoding and " + ("with sac" if agent_type.endswith('sac') else "without sac"))
    elif agent_type.startswith('dqn'):
        agent = DQNAgent(model_folder, discount=discount, sac=agent_type.endswith('sac'))
        print("Using DQN model" + (" with sac" if agent_type.endswith('sac') else ""))
    loaded_from_disk, iterations = agent.load_models_from_disk(train_models)
    if loaded_from_disk:
        print(f"Using checkpoint at iteration {iterations}")
        with open(model_folder + '/stats.pkl', 'rb') as f:
            stats = pickle.load(f)

    eval_agent: SJAgent

    if compare:
        try:
            with open(f'{compare}/state.pkl', mode='rb') as f:
                eval_state = pickle.load(f)
        except:
            raise FileNotFoundError("State file for comparison model not found")
        try:
            eval_models = importlib.import_module(f'{compare}.Models'.replace('/', '.'))
        except:
            eval_models = importlib.import_module("networks.Models")

        if eval_state.get('agent_type', 'dmc') in ('dqn', 'dqnsac', 'sac'):
            eval_agent = DQNAgent(compare, sac=eval_state['agent_type'].endwith('sac'))
        else:
            eval_agent = DMCAgent(compare, use_oracle=eval_state['oracle_duration'] > 0, dynamic_encoding=eval_state.get('dynamic_encoding', True), sac=eval_state.get('agent_type', 'dmc').endswith('sac'))
        
        loaded_eval_model, eval_iterations = eval_agent.load_models_from_disk(eval_models)
        if loaded_eval_model:
            print(f"Using evaluation checkpoint at iteration {eval_iterations}")
        
    else:
        if eval_agent_type == 'random':
            eval_agent = RandomAgent('random')
            print("Evaluating model performance using random agent...")
        elif eval_agent_type == 'strategic':
            eval_agent = StrategicAgent('strategic')
            print("Evaluating model performance using strategic agent...")
        else:
            eval_agent = InteractiveAgent('interactive')
            print("Evaluating model performance in interactive mode...")

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.ERROR)
        # logging.basicConfig(format="%(process)d %(message)s", filename=f'{model_folder}/debug.log', encoding='utf-8', level=logging.DEBUG)
    
    # Record the command used to run the script
    if not eval_only:
        with open(f'{model_folder}/command.txt', mode='w') as f:
            f.write(' '.join(sys.argv))
    
    # Load saved optimizer states
    try:
        with open(f'{model_folder}/state.pkl', mode='rb') as f:
            state = pickle.load(f)
            agent.load_optimizer_states(state)
    except:
        print("Starting new training session")
        shutil.copyfile('networks/Models.py', f'{model_folder}/Models.py')
    
    if not eval_only:
        for i in range(1 if single_process else actor_process_count):
            actor = ctx.Process(target=sampler, args=(i, agent, discount, decay_factor ** (1 / games), global_main_queue, epsilon, reuse_times, oracle_duration // actor_process_count, iterations // actor_process_count, f"{model_folder}/debug{i}.log"))
            actor.start()
            actor_processes.append(actor)
            
            print(f"Spawned process {i}")    
  
    while iterations < max_games or eval_only:
        if not eval_only:
            print(f"Training iterations {iterations}-{iterations + games}...")
            for _ in tqdm.tqdm(range(0, games, 10)):
                main_batch = global_main_queue.get()
                agent.learn_from_samples(main_batch, Stage.main_stage)
            agent.save_models_to_disk()
            print('main loss:', np.mean(agent.main_module.train_loss_history),)
            if agent.sac:
                print("Current alpha:", agent.main_module.log_alpha.exp().cpu().item())
                if isinstance(agent, DQNAgent):
                    print("value loss:", np.mean(agent.main_module.value_loss_history))
            agent.clear_loss_histories()
        
        if single_process:
            eval_sim = Simulation(
                player1=agent,
                player2=eval_agent,
                eval=True
            )
            for _ in tqdm.tqdm(range(eval_size)):
                with torch.no_grad():
                    while eval_sim.step()[0]: pass
                eval_sim.reset()
            win_counts = eval_sim.win_counts
            level_counts = eval_sim.level_counts
            opposition_points = eval_sim.opposition_points
            print(f"Average inference time: {np.mean(eval_sim.inference_times)}s")
        else:
            # eval_size = max(1, eval_size // eval_count * eval_count) # Must be multiple of eval_count
            win_counts = [0, 0] # Defenders, opponents
            level_counts = [0, 0]
            opposition_points = [[], []]
            eval_queue = ctx.Queue()
            eval_actors = []
            for i in range(min(eval_size, eval_process_count)):
                actor = ctx.Process(target=evaluator, args=(i, agent, eval_agent, max(1, eval_size // eval_process_count), eval_queue, verbose, learn_from_eval, f"{model_folder}/eval{i}.log"))
                actor.start()
                eval_actors.append(actor)
        
            with tqdm.tqdm(total=eval_size) as progress_bar:
                for i in range(eval_size):
                    win_index, opponent_index, points, levels = eval_queue.get()
                    win_counts[win_index] += 1
                    level_counts[win_index] += levels
                    opposition_points[opponent_index].append(points)
                    progress_bar.update(1)
                
        
            for a in eval_actors:
                a.kill()

        print('Win counts:', win_counts, 'level counts:', level_counts)
        print("Average opposition points:", np.mean(opposition_points[0]), np.mean(opposition_points[1]))
        
        iterations += games

        if not eval_only:
            stats.append({
                "iterations": iterations,
                "win_counts": win_counts[0] / sum(win_counts),
                "level_counts": level_counts[0] / sum(level_counts),
                "avg_points": [np.mean(opposition_points[0]), np.mean(opposition_points[1])]
            })
            with open(f'{model_folder}/stats.pkl', mode='w+b') as f:
                pickle.dump(stats, f)
            with open(f'{model_folder}/state.pkl', mode='w+b') as f:
                pickle.dump({
                    'agent_type': agent_type,
                    'iterations': iterations,
                    **agent.optimizer_states(),
                    'oracle_duration': oracle_duration_input,
                    'dynamic_encoding': dynamic_encoding
                }, f)
        else:
            break
    
    for c in ctx.active_children():
        c.kill()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train loop')
    parser.add_argument('--agent-type', type=str, default='dmc', choices=['dmc', 'dqn', 'dqnsac', 'dmcsac'])
    parser.add_argument('--games', type=int, default=500)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-size', type=int, default=300)
    parser.add_argument('--model-folder', type=str, default='pretrained')
    parser.add_argument('--compare', type=str, default='')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--decay-factor', type=float, default=1.2)
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--single-process', action='store_true')
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--eval-agent', type=str, default='random', choices=['random', 'interactive', 'strategic'])
    parser.add_argument('--learn-from-eval', action='store_true')
    parser.add_argument('--reuse-times', type=int, default=0)
    parser.add_argument('--oracle-duration', type=int, default=0)
    parser.add_argument('--max-games', type=int, default=500000)
    parser.add_argument('--static-encoding', action='store_true')
    parser.add_argument('--actor-processes', type=int, default=6)
    parser.add_argument('--eval-processes', type=int, default=7)
    args = parser.parse_args()
    train(args.agent_type, args.games, args.model_folder, args.eval_only, args.eval_size, args.compare, args.discount, args.decay_factor, args.verbose, args.random_seed, args.single_process, args.epsilon, args.tau, args.eval_agent, args.learn_from_eval, args.reuse_times, args.oracle_duration, args.max_games, not args.static_encoding, args.actor_processes, args.eval_processes)