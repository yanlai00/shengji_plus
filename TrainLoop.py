import logging
import pickle
import random
import shutil
import torch, os
from Simulation import Simulation
from agents.RLAgents import *
import argparse
import tqdm
import numpy as np

def train(games: int, model_folder: str, eval_only: bool, eval_size: int, compare: str = None, discount=0.99, verbose=False):
    os.makedirs(model_folder, exist_ok=True)
    torch.manual_seed(0)
    random.seed(2)
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
    
    train_sim = Simulation(
        main_agent=main_agent,
        declare_agent=declare_agent,
        kitty_agent=kitty_agent,
        chaodi_agent=chaodi_agent,
        enable_combos=False,
        eval_main=eval_main,
        eval_declare=eval_declare,
        eval_kitty=eval_kitty,
        eval_chaodi=eval_chaodi,
        discount=discount,
        eval=eval_only,
    )

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

    while True:
        if not eval_only:
            print(f"Training games {iterations * games}-{(iterations+1) * games}...")
        for i in tqdm.tqdm(range(games)):
            with torch.no_grad():
                while train_sim.step()[0]: pass
            if not eval_only:
                if (i+1) % 5 == 0:
                    train_sim.backprop()
                    for agent in (declare_agent, kitty_agent, chaodi_agent, main_agent):
                        torch.save(agent.model.state_dict(), f'{model_folder}/{agent.name}.pt')
            train_sim.reset()

        if eval_only:
            print('Win counts:', train_sim.win_counts)
            print("Average opposition points:", np.mean(train_sim.opposition_points[0]), np.mean(train_sim.opposition_points[1]))
            break
        
        if not eval_only:
            print("Evaluating model performance...")
            eval_sim = Simulation(
                main_agent=main_agent,
                declare_agent=declare_agent,
                kitty_agent=kitty_agent,
                chaodi_agent=chaodi_agent,
                enable_combos=False,
                eval=True
            )
            for _ in tqdm.tqdm(range(eval_size)):
                with torch.no_grad():
                    while eval_sim.step()[0]: pass
                eval_sim.reset()
            print('Win counts:', eval_sim.win_counts)
            print("Average opposition points:", np.mean(eval_sim.opposition_points[0]), np.mean(eval_sim.opposition_points[1]))
        
        iterations += 1
        stats.append({
            "iterations": iterations * games,
            "win_counts": eval_sim.win_counts[0] / sum(eval_sim.win_counts),
            "avg_points": [np.mean(eval_sim.opposition_points[0]), np.mean(eval_sim.opposition_points[1])]
        })
        with open(f'{model_folder}/stats.pkl', mode='w+b') as f:
            pickle.dump(stats, f)
        with open(f'{model_folder}/state.pkl', mode='w+b') as f:
            pickle.dump({ 'iterations': iterations }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train loop')
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-size', type=int, default=300)
    parser.add_argument('--model-folder', type=str, default='pretrained')
    parser.add_argument('--compare', type=str, default='')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--discount', type=float, default=0.99)
    args = parser.parse_args()
    train(args.games, args.model_folder, args.eval_only, args.eval_size, args.compare, args.discount, args.verbose)