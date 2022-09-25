# A class that simulates a game
import random
from agents.Agent import SJAgent
from env.utils import AbsolutePosition 
from env.Game import Game

class Simulation:
    def __init__(self, agent1: SJAgent, agent2: SJAgent) -> None:
        self.agents = {
            AbsolutePosition.NORTH: agent1,
            AbsolutePosition.SOUTH: agent1,
            AbsolutePosition.WEST: agent2,
            AbsolutePosition.EAST: agent2,
        }
        self.game_engine = Game(enable_chaodi=True, enable_combos=True)
        self.current_player = None

    def step(self):
        "Step the game and return whether the game is still ongoing."
        if not self.game_engine.game_started:
            self.current_player = self.game_engine.start_game()
            return True, None
        
        if not self.game_engine.game_ended:
            observation = self.game_engine.get_observation(self.current_player)
            action = self.agents[self.current_player].act(observation)
            self.current_player, reward = self.game_engine.run_action(action, self.current_player)
            return True, action
        else:
            return False, None

    
    def reset(self):
        self.game_engine = Game()