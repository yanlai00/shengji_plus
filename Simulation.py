# A class that simulates a game
import logging
import random
from typing import Deque, Dict, List, Tuple
from agents.Agent import RandomAgent, SJAgent
from env.Observation import Observation
from env.utils import AbsolutePosition 
from env.Game import Game, Stage
from env.Actions import *
from collections import deque

FIXED_DECK = CardSet.new_deck()[1]

class Simulation:
    def __init__(self, main_agent: SJAgent, declare_agent: SJAgent, kitty_agent: SJAgent, chaodi_agent: SJAgent = None, discount=0.99, enable_combos=False, eval=False, eval_main: SJAgent = None, eval_declare: SJAgent = None, eval_kitty: SJAgent = None, eval_chaodi: SJAgent = None, epsilon=0.98) -> None:
        "If eval = True, use random agents for East and West."

        self.main_agent = main_agent
        self.declare_agent = declare_agent
        self.kitty_agent = kitty_agent
        self.chaodi_agent = chaodi_agent
        self.game_engine = Game(
            dominant_rank=random.randint(2, 14),
            dealer_position=AbsolutePosition.random() if random.random() > 0.5 else None,
            enable_chaodi=chaodi_agent is not None,
            enable_combos=enable_combos)

        self.current_player = None
        self.discount = discount
        self.win_counts = [0, 0] # index 0 = wins of N and S; index 1 = wins of W and E
        self.level_counts = [0, 0]
        self.opposition_points = [[], []] # index 0 is the opposition points for N and S;
        self.eval_mode = eval
        self.random_agent = RandomAgent('random')
        self.eval_main = eval_main
        self.eval_declare = eval_declare
        self.eval_kitty = eval_kitty
        self.eval_chaodi = eval_chaodi
        self.epsilon = epsilon

        # (state, action, reward) tuples for each player during the main stage of the game
        self._main_history_per_player: Dict[AbsolutePosition, List[Tuple[Observation, Action, float]]] = {
            AbsolutePosition.NORTH: [],
            AbsolutePosition.SOUTH: [],
            AbsolutePosition.WEST: [],
            AbsolutePosition.EAST: []
        }
        self.main_history: List[Tuple[Observation, Action, float]] = []

        # (state, action, reward) tuples for each player who got to bury the kitty
        self._kitty_history_per_player: Dict[AbsolutePosition, List[Tuple[Observation, Action, float]]] = {
            AbsolutePosition.NORTH: [],
            AbsolutePosition.SOUTH: [],
            AbsolutePosition.WEST: [],
            AbsolutePosition.EAST: []
        }
        self.kitty_history: List[Tuple[Observation, Action, float]] = []

        # (state, action, reward) tuples for each player who declared a trump suite
        self._declaration_history_per_player: Dict[AbsolutePosition, List[Tuple[Observation, Action, float]]] = {
            AbsolutePosition.NORTH: [],
            AbsolutePosition.SOUTH: [],
            AbsolutePosition.WEST: [],
            AbsolutePosition.EAST: []
        }
        self.declaration_history: List[Tuple[Observation, Action, float]] = []

        # (state, action, reward) tuples for each player who chaodied.
        self._chaodi_history_per_player: Dict[AbsolutePosition, List[Tuple[Observation, Action, float]]] = {
            AbsolutePosition.NORTH: [],
            AbsolutePosition.SOUTH: [],
            AbsolutePosition.WEST: [],
            AbsolutePosition.EAST: []
        }
        self.chaodi_history: List[Tuple[Observation, Action, float]] = []


    def step(self):
        "Step the game and return whether the game is still ongoing."
        if not self.game_engine.game_started:
            self.current_player = self.game_engine.start_game()
            return True, None
        
        if not self.game_engine.game_ended:
            observation = self.game_engine.get_observation(self.current_player)
            if self.eval_mode and observation.position in [AbsolutePosition.EAST, AbsolutePosition.WEST]:
                if self.game_engine.stage == Stage.declare_stage:
                    action = (self.eval_declare or self.random_agent).act(observation)
                elif self.game_engine.stage == Stage.kitty_stage:
                    action = (self.eval_kitty or self.random_agent).act(observation)
                elif self.game_engine.stage == Stage.chaodi_stage:
                    action = (self.eval_chaodi or self.random_agent).act(observation)
                else:
                    action = (self.eval_main or self.random_agent).act(observation)
            else:
                # Depending on the stage of the game, we use different agents to calculate an action
                if self.game_engine.stage == Stage.declare_stage:
                    action = self.declare_agent.act(observation, epsilon=not self.eval_mode and self.epsilon)
                elif self.game_engine.stage == Stage.kitty_stage:
                    action = self.kitty_agent.act(observation, epsilon=not self.eval_mode and self.epsilon)
                elif self.game_engine.stage == Stage.chaodi_stage:
                    action = self.chaodi_agent.act(observation, epsilon=not self.eval_mode and self.epsilon)
                else:
                    action = self.main_agent.act(observation, epsilon=not self.eval_mode and self.epsilon)
            
            last_stage = self.game_engine.stage
            last_player = self.current_player
            self.current_player, reward = self.game_engine.run_action(action, self.current_player)
            
            # Collect the observation, action and reward (rewards will be updated after the game finished)
            if last_stage == Stage.declare_stage:
                self._declaration_history_per_player[last_player].append((observation, action, reward))
            elif last_stage == Stage.kitty_stage:
                self._kitty_history_per_player[last_player].append((observation, action, reward))
            elif last_stage == Stage.chaodi_stage:
                self._chaodi_history_per_player[last_player].append((observation, action, reward))
            else:
                self._main_history_per_player[last_player].append((observation, action, reward))
            
            # Helper function that determines if a player is on the defender side or the opponent side
            def is_defender(player: AbsolutePosition):
                return player == self.game_engine.dealer_position or player.next_position.next_position == self.game_engine.dealer_position
            
            # If we reached the end of the game, propagate rewards for all states and actions.
            if self.game_engine.game_ended:
                
                # Determine who won and update the win count
                if self.game_engine.opponent_points >= 80:
                    if self.game_engine.dealer_position in [AbsolutePosition.NORTH, AbsolutePosition.SOUTH]:
                        self.win_counts[1] += 1
                        self.level_counts[1] += self.game_engine.final_opponent_reward
                    else:
                        self.win_counts[0] += 1
                        self.level_counts[0] += self.game_engine.final_opponent_reward
                else:
                    if self.game_engine.dealer_position in [AbsolutePosition.NORTH, AbsolutePosition.SOUTH]:
                        self.win_counts[0] += 1
                        self.level_counts[0] += self.game_engine.final_defender_reward
                    else:
                        self.win_counts[1] += 1
                        self.level_counts[1] += self.game_engine.final_defender_reward
                
                if self.game_engine.dealer_position in [AbsolutePosition.NORTH, AbsolutePosition.SOUTH]:
                    self.opposition_points[1].append(self.game_engine.opponent_points)
                else:
                    self.opposition_points[0].append(self.game_engine.opponent_points)

                logging.debug("Data for current round:")
                logging.debug('Main history: ' + str({k.value: len(v) for k, v in self._main_history_per_player.items()}))
                logging.debug('Declaration history: ' + str({k.value: len(v) for k, v in self._declaration_history_per_player.items()}))
                logging.debug('Kitty history: ' + str({k.value: len(v) for k, v in self._kitty_history_per_player.items()}))
                logging.debug('Chaodi history: ' + str({k.value: len(v) for k, v in self._chaodi_history_per_player.items()}))

                if not self.eval_mode:
                    for position in ['N', 'W', 'S', 'E']:
                        # For kitty action, the reward is not discounted because each move is equally important
                        for i in range(len(self._kitty_history_per_player[position])):
                            ob, ac, rw = self._kitty_history_per_player[position][i]
                            if is_defender(ob.position):
                                self._kitty_history_per_player[position][i] = (ob, ac, rw + self.game_engine.final_defender_reward)
                            else:
                                self._kitty_history_per_player[position][i] = (ob, ac, rw + self.game_engine.final_opponent_reward)
                            self.kitty_history.append(self._kitty_history_per_player[position][i])
                        self._kitty_history_per_player[position].clear()

                        # For declaration, the reward is also not discounted for the same reason
                        for i in range(len(self._declaration_history_per_player[position])):
                            ob, ac, rw = self._declaration_history_per_player[position][i]
                            if is_defender(ob.position):
                                self._declaration_history_per_player[position][i] = (ob, ac, rw + self.game_engine.final_defender_reward)
                            else:
                                self._declaration_history_per_player[position][i] = (ob, ac, rw + self.game_engine.final_opponent_reward)
                            self.declaration_history.append(self._declaration_history_per_player[position][i])
                        self._declaration_history_per_player[position].clear()

                        # For chaodi, the reward is not discounted also
                        for i in range(len(self._chaodi_history_per_player[position])):
                            ob, ac, rw = self._chaodi_history_per_player[position][i]
                            if is_defender(ob.position):
                                self._chaodi_history_per_player[position][i] = (ob, ac, rw + self.game_engine.final_defender_reward)
                            else:
                                self._chaodi_history_per_player[position][i] = (ob, ac, rw + self.game_engine.final_opponent_reward)
                            self.chaodi_history.append(self._chaodi_history_per_player[position][i])
                        self._chaodi_history_per_player[position].clear()
                            
                        # For main history, the reward is slightly discounted
                        for i in reversed(range(len(self._main_history_per_player[position]))):
                            ob, ac, rw = self._main_history_per_player[position][i]

                            # Add reward from next time step if there is one, otherwise assign final reward
                            if i + 1 < len(self._main_history_per_player[position]):
                                rw += self.discount * self._main_history_per_player[position][i+1][2]
                            else:
                                rw += self.game_engine.final_defender_reward if is_defender(ob.position) else self.game_engine.final_opponent_reward

                            if is_defender(ob.position):
                                if self.game_engine.points_per_round[i] >= 0:
                                    rw += self.game_engine.points_per_round[i] / 40 # Defenders are only moderately happy when escaping points
                                else:
                                    rw += self.game_engine.points_per_round[i] / 40 # Defenders should care a lot about losing points
                                self._main_history_per_player[position][i] = (ob, ac, rw)
                            else:
                                if self.game_engine.points_per_round[i] <= 0:
                                    rw -= self.game_engine.points_per_round[i] / 40 # Opponents are happier when earning points
                                else:
                                    rw -= self.game_engine.points_per_round[i] / 40 # Opponents are not so sad when they lose points
                                self._main_history_per_player[position][i] = (ob, ac, rw)
                            self.main_history.append(self._main_history_per_player[position][i])
                        self._main_history_per_player[position].clear()
                    # for history in [self.main_history, self.declaration_history, self.kitty_history, self.chaodi_history]:
                    #     random.shuffle(history)

            return True, action
        else:
            return False, None

    
    def reset(self, reuse_old_deck=False):
        old_deck = self.game_engine.card_list
        self.game_engine = Game(
            dominant_rank=random.randint(2, 14),
            dealer_position=AbsolutePosition.random() if random.random() > 0.5 else None,
            enable_chaodi=self.game_engine.enable_chaodi,
            enable_combos=self.game_engine.enable_combos)
        self.main_history.clear()
        self.declaration_history.clear()
        self.chaodi_history.clear()
        self.kitty_history.clear()

        if reuse_old_deck:
            self.game_engine.deck = iter(old_deck)
    
    def backprop(self):
        self.declare_agent.learn_from_samples(self.declaration_history)
        self.kitty_agent.learn_from_samples(self.kitty_history)
        self.chaodi_agent.learn_from_samples(self.chaodi_history)
        self.main_agent.learn_from_samples(self.main_history)