import logging
import torch, os
from Simulation import Simulation
from agents.RLAgents import *
import argparse
import tqdm

def train(games: int, eval: bool):
    os.makedirs('pretrained', exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    declare_model = DeclarationModel()
    if os.path.exists('pretrained/declare.pt'):
        declare_agent = declare_model.load_state_dict(torch.load('pretrained/declare.pt', map_location=device))
    declare_agent = DeclareAgent('declare', declare_model)
    
    kitty_model = KittyModel()
    if os.path.exists('pretrained/kitty.pt'):
        kitty_model.load_state_dict(torch.load('pretrained/kitty.pt', map_location=device))
    kitty_agent = KittyAgent('kitty', kitty_model)
    
    chaodi_model = ChaodiModel()
    if os.path.exists('pretrained/chaodi.pt'):
        chaodi_model.load_state_dict(torch.load('pretrained/chaodi.pt', map_location=device))
    chaodi_agent = ChaodiAgent('chaodi', chaodi_model)

    main_model = MainModel()
    if os.path.exists('pretrained/main.pt'):
        main_model.load_state_dict(torch.load('pretrained/main.pt', map_location=device))
    main_agent = MainAgent('main', main_model)
    
    train_sim = Simulation(
        main_agent=main_agent,
        declare_agent=declare_agent,
        kitty_agent=kitty_agent,
        chaodi_agent=chaodi_agent,
        eval=eval
    )

    logging.getLogger().setLevel(logging.WARNING)
    
    for i in tqdm.tqdm(range(games)):
        while train_sim.step()[0]: pass
        if not eval:
            train_sim.backprop()
            for agent in (declare_agent, kitty_agent, chaodi_agent, main_agent):
                torch.save(agent.model.state_dict(), f'pretrained/{agent.name}.pt')
        train_sim.reset()

    print(train_sim.win_counts)
    

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train loop')
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    train(args.games, args.eval)