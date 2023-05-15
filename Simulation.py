# A class that simulates a game
from datetime import datetime
import logging
import random
from typing import Deque, Dict, List, Tuple
from agents.RandomAgent import RandomAgent
from agents.Agent import SJAgent
from agents.DQNAgent import DQNAgent
from env.Observation import Observation
from env.utils import AbsolutePosition 
from env.Game import Game, Stage
from env.Actions import *
from collections import deque

class Simulation:
    def __init__(self, player1: SJAgent, player2: SJAgent=None, discount=0.99, eval=False, epsilon=0.02, learn_from_eval=False, oracle_duration=0, game_count=0) -> None:
        "If eval = True, use random agents for East and West."

        self.oracle_duration = oracle_duration
        self.game_count = game_count

        self.game_engine = Game(
            dealer_position=AbsolutePosition.random() if random.random() > 0.5 else None,
            oracle_value=self.oracle_value,
        )

        self.current_player = None
        self.discount = discount
        self.win_counts = [0, 0] # index 0 = wins of N and S; index 1 = wins of W and E
        self.level_counts = [0, 0]
        self.opposition_points = [[], []] # index 0 is the opposition points for N and S;
        self.eval_mode = eval
        self.learn_from_eval = learn_from_eval
        self.player1 = player1 # only this player is being trained
        self.player2 = player2
        self.epsilon = epsilon

        # (state, action, reward) tuples for each player during the main stage of the game
        self._main_history_per_player: Dict[AbsolutePosition, List[Tuple[Observation, Action, float]]] = {
            AbsolutePosition.NORTH: [],
            AbsolutePosition.SOUTH: [],
            AbsolutePosition.WEST: [],
            AbsolutePosition.EAST: []
        }
        self.main_history: List[Tuple[Observation, Action, float, Observation]] = []

        # (state, action, reward) tuples for each player who declared a trump suit
        self._declaration_history_per_player: Dict[AbsolutePosition, List[Tuple[Observation, Action, float]]] = {
            AbsolutePosition.NORTH: [],
            AbsolutePosition.SOUTH: [],
            AbsolutePosition.WEST: [],
            AbsolutePosition.EAST: []
        }
        self.declaration_history: List[Tuple[Observation, Action, float, Observation]] = []

        self.action_entropy_per_player = {
            AbsolutePosition.NORTH: [],
            AbsolutePosition.SOUTH: [],
            AbsolutePosition.WEST: [],
            AbsolutePosition.EAST: []
        }

        self.cumulative_rewards = not isinstance(player1, DQNAgent) # Whether to compute cumulative rewards at the end

        self.inference_times = []

    def step(self):
        "Step the game and return whether the game is still ongoing."
        if not self.game_engine.game_started:
            self.current_player = self.game_engine.start_game()
            return True, None
        
        if not self.game_engine.game_ended:
            observation = self.game_engine.get_observation(self.current_player)
            if self.eval_mode and observation.position in [AbsolutePosition.EAST, AbsolutePosition.WEST]:
                # In evaluation mode, player2 plays EAST and WEST
                action, _, _ = self.player2.act(observation, training=False)
            else:
                # In training mode, player1 plays all 4 positions
                if self.eval_mode:
                    start = datetime.now().timestamp()
                action, max_prob, entropy = self.player1.act(observation, epsilon=not self.eval_mode and self.epsilon, training=not self.eval_mode)
                if self.eval_mode:
                    self.inference_times.append(datetime.now().timestamp() - start)
            
            last_stage = self.game_engine.stage
            last_player = self.current_player
            self.current_player, reward = self.game_engine.run_action(action, self.current_player)
            
            # Collect the observation, action and reward if training(rewards will be updated after the game finished)
            if not self.eval_mode or self.learn_from_eval and self.current_player in (AbsolutePosition.NORTH, AbsolutePosition.SOUTH):
                self._main_history_per_player[last_player].append((observation, action, reward))
                self.action_entropy_per_player[last_player].append((max_prob, entropy))
    
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

                position_list = []
                if self.learn_from_eval:
                    position_list = ['N', 'S']
                elif not self.eval_mode:
                    position_list = ['N', 'W', 'S', 'E']
                else:
                    return True, action # skip action collection process if in eval mode

                for position in position_list:

                    # For declaration, the reward is also not discounted for the same reason
                    for i in range(len(self._declaration_history_per_player[position])):
                        ob, ac, rw = self._declaration_history_per_player[position][i]
                        if is_defender(ob.position):
                            self._declaration_history_per_player[position][i] = (ob, ac, rw + self.game_engine.final_defender_reward)
                        else:
                            self._declaration_history_per_player[position][i] = (ob, ac, rw + self.game_engine.final_opponent_reward)
                        self.declaration_history.append(self._declaration_history_per_player[position][i])
                    self._declaration_history_per_player[position].clear()
                        
                    # For main history, the reward is slightly discounted
                    for i in reversed(range(len(self._main_history_per_player[position]))):
                        ob, ac, rw = self._main_history_per_player[position][i]

                        # Add reward from next time step if there is one, otherwise assign final reward
                        if i + 1 < len(self._main_history_per_player[position]):
                            if self.cumulative_rewards:
                                rw += self.discount * self._main_history_per_player[position][i+1][2]
                        else:
                            rw += self.game_engine.final_defender_reward if is_defender(ob.position) else self.game_engine.final_opponent_reward

                        # Add rewards for points earned / lost in current round
                        if is_defender(ob.position):
                            if self.game_engine.points_per_round[i] >= 0:
                                rw += self.game_engine.points_per_round[i] / 80 # Defenders are only moderately happy when escaping points
                            else:
                                rw += self.game_engine.points_per_round[i] / 60 # Defenders should care a lot about losing points
                        else:
                            if self.game_engine.points_per_round[i] <= 0:
                                rw -= self.game_engine.points_per_round[i] / 60 # Opponents are happier when earning points
                            else:
                                rw -= self.game_engine.points_per_round[i] / 80 # Opponents are not so sad when they lose points
                        
                        next_ob = None
                        if getattr(self.player1, 'sac', False):
                            next_ob = self._main_history_per_player[position][i + 1][0] if i+1 < len(self._main_history_per_player[position]) else None
                        self._main_history_per_player[position][i] = (ob, ac, rw, (
                            next_ob,
                            self.action_entropy_per_player[position][i][1],
                            self.action_entropy_per_player[position][i+1][1] if i+1 < len(self._main_history_per_player[position]) else None # entropy of next action distribution
                        ))
                        
                        self.main_history.append(self._main_history_per_player[position][i])
                    self._main_history_per_player[position] = []

            return True, action
        else:
            self.game_count += 1
            return False, None

    @property
    def oracle_value(self):
        if self.oracle_duration == 0:
            return 0.0
        else:
            return max(0, (self.oracle_duration - self.game_count) / self.oracle_duration)
    
    def reset(self, reuse_old_deck=False):
        old_deck = self.game_engine.card_list
        old_dealer = self.game_engine.dealer_position
        self.game_engine = Game(
            dealer_position=AbsolutePosition.random() if random.random() > 0.5 else None,
            oracle_value=self.oracle_value,
        )
        self.main_history.clear()
        self.declaration_history.clear()

        # If reuse_old_deck, then re-play the game using the same hands
        if reuse_old_deck:
            self.game_engine.deck = iter(old_deck)
            self.game_engine.dealer_position = old_dealer
        