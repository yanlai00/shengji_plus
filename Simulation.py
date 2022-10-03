# A class that simulates a game
import random
from typing import Dict, List, Tuple
from agents.Agent import SJAgent
from env.Observation import Observation
from env.utils import AbsolutePosition 
from env.Game import Game, Stage
from env.Actions import *

class Simulation:
    def __init__(self, main_agent: SJAgent, declare_agent: SJAgent, kitty_agent: SJAgent, chaodi_agent: SJAgent = None, discount=0.98, enable_combos=False) -> None:
        self.main_agent = main_agent
        self.declare_agent = declare_agent
        self.kitty_agent = kitty_agent
        self.chaodi_agent = chaodi_agent
        self.game_engine = Game(enable_chaodi=chaodi_agent is not None, enable_combos=enable_combos)
        self.current_player = None
        self.discount = discount

        # (state, action, reward) tuples for each player during the main stage of the game
        self.main_history: Dict[AbsolutePosition, List[Tuple[Observation, Action, float]]] = {
            AbsolutePosition.NORTH: [],
            AbsolutePosition.SOUTH: [],
            AbsolutePosition.WEST: [],
            AbsolutePosition.EAST: []
        }

        # (state, action, reward) tuples for each player who got to bury the kitty
        self.place_kitty_history: Dict[AbsolutePosition, List[Tuple[Observation, Action, float]]] = {
            AbsolutePosition.NORTH: [],
            AbsolutePosition.SOUTH: [],
            AbsolutePosition.WEST: [],
            AbsolutePosition.EAST: []
        }

        # (state, action, reward) tuples for each player who declared a trump suite
        self.declaration_history: Dict[AbsolutePosition, List[Tuple[Observation, Action, float]]] = {
            AbsolutePosition.NORTH: [],
            AbsolutePosition.SOUTH: [],
            AbsolutePosition.WEST: [],
            AbsolutePosition.EAST: []
        }

        # (state, action, reward) tuples for each player who chaodied.
        self.chaodi_history: Dict[AbsolutePosition, List[Tuple[Observation, Action, float]]] = {
            AbsolutePosition.NORTH: [],
            AbsolutePosition.SOUTH: [],
            AbsolutePosition.WEST: [],
            AbsolutePosition.EAST: []
        }

    def step(self):
        "Step the game and return whether the game is still ongoing."
        if not self.game_engine.game_started:
            self.current_player = self.game_engine.start_game()
            return True, None
        
        if not self.game_engine.game_ended:
            observation = self.game_engine.get_observation(self.current_player)
            
            # Depending on the stage of the game, we use different agents to calculate an action
            if self.game_engine.stage == Stage.declare_stage:
                action = self.declare_agent.act(observation)
            elif self.game_engine.stage == Stage.kitty_stage:
                action = self.kitty_agent.act(observation)
            elif self.game_engine.stage == Stage.chaodi_stage:
                action = self.chaodi_agent.act(observation)
            else:
                action = self.main_agent.act(observation)
            
            last_stage = self.game_engine.stage
            last_player = self.current_player
            self.current_player, reward = self.game_engine.run_action(action, self.current_player)
            
            # Collect the observation, action and reward (rewards will be updated after the game finished)
            if last_stage == Stage.declare_stage:
                self.declaration_history[last_player].append((observation, action, reward))
            elif last_stage == Stage.kitty_stage:
                self.place_kitty_history[last_player].append((observation, action, reward))
            elif last_stage == Stage.chaodi_stage:
                self.chaodi_history[last_player].append((observation, action, reward))
            else:
                self.main_history[last_player].append((observation, action, reward))
            
            # Helper function that determines if a player is on the defender side or the opponent side
            def is_defender(player: AbsolutePosition):
                return player == self.game_engine.dealer_position or player.next_position.next_position == self.game_engine.dealer_position
            
            # If we reached the end of the game, propagate rewards for all states and actions.
            if self.game_engine.game_ended:
                for position in ['N', 'W', 'S', 'E']:
                    # For kitty action, the reward is not discounted because each move is equally important
                    for i in range(len(self.place_kitty_history[position])):
                        ob, ac, rw = self.place_kitty_history[position][i]
                        if is_defender(ob.position):
                            self.place_kitty_history[position][i] = (ob, ac, rw + self.game_engine.final_defender_reward)
                        else:
                            self.place_kitty_history[position][i] = (ob, ac, rw + self.game_engine.final_opponent_reward)
                    
                    # For declaration, the reward is also not discounted for the same reason
                    for i in range(len(self.declaration_history[position])):
                        ob, ac, rw = self.declaration_history[position][i]
                        if is_defender(ob.position):
                            self.declaration_history[position][i] = (ob, ac, rw + self.game_engine.final_defender_reward)
                        else:
                            self.declaration_history[position][i] = (ob, ac, rw + self.game_engine.final_opponent_reward)
                    
                    # For chaodi, the reward is not discounted also
                    for i in range(len(self.chaodi_history[position])):
                        ob, ac, rw = self.chaodi_history[position][i]
                        if is_defender(ob.position):
                            self.chaodi_history[position][i] = (ob, ac, rw + self.game_engine.final_defender_reward)
                        else:
                            self.chaodi_history[position][i] = (ob, ac, rw + self.game_engine.final_opponent_reward)
                        
                    # For main history, the reward is slightly discounted
                    for i in range(len(self.main_history[position])):
                        ob, ac, rw = self.main_history[position][i]
                        discount_factor = self.discount ** (len(self.main_history[position]) - i - 1)
                        if is_defender(ob.position):
                            self.main_history[position][i] = (ob, ac, rw + self.game_engine.final_defender_reward * discount_factor)
                        else:
                            self.main_history[position][i] = (ob, ac, rw + self.game_engine.final_opponent_reward * discount_factor)

            return True, action
        else:
            print("Data collection status:")
            print('Main history:', {k.value: len(v) for k, v in self.main_history.items()})
            print('Declaration history:', {k.value: len(v) for k, v in self.declaration_history.items()})
            print('Kitty history:', {k.value: len(v) for k, v in self.place_kitty_history.items()})
            print('Chaodi history:', {k.value: len(v) for k, v in self.chaodi_history.items()})
            return False, None

    
    def reset(self):
        self.game_engine = Game()