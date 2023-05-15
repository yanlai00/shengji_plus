# Need to rewrite this file

from .Agent import SJAgent, StageModule
from typing import List, Tuple
import sys
import random
from env.CardSet import CardSet, MoveType, TrumpSuit

sys.path.append('.')
from env.Actions import *
from env.utils import ORDERING_INDEX, Stage, softmax, CardSuit
from env.Observation import Observation
from networks.Models import *

class MainModule(StageModule):
    def act(self, obs: Observation, epsilon=None, training=True):
        if obs.leads_current_round:
            optimal_actions = []
            second_best_actions = []
            other_actions = []
            for action in obs.actions:
                d = action.move.cardset.decompose(obs.dominant_suit, obs.dominant_rank)
                
                # For simplicity, this agent doesn't play combos
                if len(d) > 1:
                    # If the combo is unbeatable (unless using trump cards)
                    if CardSet.is_bigger_than(obs.unplayed_cards, action.move, obs.dominant_suit, obs.dominant_rank) is None:
                        # If the combo contains a pair of a tractor, then it's probably a good move to play
                        if not all([isinstance(component, MoveType.Single) for component in d]):
                            optimal_actions.append(action)
                        else:
                            second_best_actions.append(action)
                    print(action.move)
                elif action.move.cardset.count_suit(CardSuit.TRUMP, obs.dominant_suit, obs.dominant_rank) == 0:
                    min_rank = d[0].cardset.min_rank(obs.dominant_suit, obs.dominant_rank)
                    if min_rank >= 8 and isinstance(d[0], MoveType.Single) and (min_rank == 14 or obs.dominant_rank == 14 and min_rank == 13):
                        optimal_actions.append(action)
                    elif d[0].cardset.min_rank(obs.dominant_suit, obs.dominant_rank) > 10:
                        second_best_actions.append(action)
                    else:
                        other_actions.append(action)
                else:
                    if action.move.cardset.total_points() == 0:
                        optimal_actions.append(action)
                    else:
                        other_actions.append(action)
                
                if optimal_actions:
                    return random.choice(optimal_actions), None, None
                elif second_best_actions:
                    return random.choice(second_best_actions), None, None
                else:
                    return other_actions[0], None, None
                        
        else:
            current_round_points = 0
            dominant_suit = obs.declaration.suit if obs.declaration else TrumpSuit.NT
            for h in obs.round_history[-1][1]:
                current_round_points += h.total_points()
            
            good_action_choices = []
            other_action_choices = []
            lead_index = CardSet.round_winner(obs.round_history[-1][1], dominant_suit, obs.dominant_rank)
            teammate_is_biggest = len(obs.round_history[-1][1]) - lead_index == 2
            
            for action in obs.actions:
                if not isinstance(action, FollowAction): continue

                is_dominating = CardSet.round_winner(obs.round_history[-1][1] + [action.cardset], dominant_suit, obs.dominant_rank) == len(obs.round_history[-1][1])
                if (current_round_points > 0 or len(obs.round_history[-1][1]) == 2) and not teammate_is_biggest:
                    if is_dominating:
                        good_action_choices.append(action)
                    else:
                        other_action_choices.append(action)
                else:
                    good_action_choices.append(action)
            
            if good_action_choices:
                return random.choice(good_action_choices), None, None
            else:
                return random.choice(other_action_choices), None, None

class StrategicAgent(SJAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.main_module = MainModule()

