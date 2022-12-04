
import logging
import random
import sys
from typing import List, Tuple

sys.path.append('.')
from env.Observation import Observation
from env.Actions import Action, ChaodiAction, DeclareAction, DontChaodiAction, DontDeclareAction, FollowAction, LeadAction, PlaceKittyAction
from env.utils import AbsolutePosition, CardSuite, Declaration, Stage, TrumpSuite, get_rank, get_suite
from env.CardSet import CardSet, MoveType

from env.utils import AbsolutePosition

class SJAgent:
    def __init__(self, name: str) -> None:
        self.name = name
    
    def act(self, obs: Observation, training=True):
        raise NotImplementedError
    
    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float]]):
        raise NotImplementedError

class RandomAgent(SJAgent):
    def act(self, obs: Observation, **kwargs):
        non_combo: List[Action] = []
        combo: List[Action] = []
        others: List[Action] = []
        for a in obs.actions:
            if isinstance(a, LeadAction):
                if isinstance(a.move, MoveType.Combo):
                    combo.append(a)
                else:
                    non_combo.append(a)
            else:
                others.append(a)
        
        if others:
            return random.choice(others)
        elif not non_combo:
            return random.choice(combo)
        elif not combo:
            return random.choice(non_combo)
        else:
            return random.choice(combo) if random.random() > 0.9 else random.choice(non_combo)

class InteractiveAgent(SJAgent):
    def act(self, obs: Observation, **kwargs):
        if len(obs.actions) == 1 and obs.stage == Stage.declare_stage: return obs.actions[0]
        print(f'current hand ({obs.position.value}):', obs.hand)
        for i, a in enumerate(obs.actions):
            print(f'{i:>5}\t{a}')
        while True:
            idx = input("Enter action index: ")
            try:
                return obs.actions[int(idx)]
            except:
                continue
            

class StrategicAgent(SJAgent):
    def act(self, obs: Observation, **kwargs):
        dominant_suite = obs.declaration.suite if obs.declaration else TrumpSuite.XJ
        if obs.stage == Stage.declare_stage:
            best_declare_option = None
            best_relative_count = 0
            for declare_action in obs.actions:
                if isinstance(declare_action, DeclareAction) and declare_action.declaration.suite not in ['XJ', 'DJ']:
                    trump_count = obs.hand.count_suite(CardSuite.TRUMP, declare_action.declaration.suite, obs.dominant_rank)
                    if trump_count > best_relative_count:
                        best_relative_count = trump_count
                        best_declare_option = declare_action
            if best_relative_count > 5 and best_relative_count / obs.hand.size >= 0.4:
                return best_declare_option
            else:
                return DontDeclareAction()
        if obs.stage == Stage.kitty_stage:
            best_actions: List[Tuple[int, Action]] = []
            second_best_actions: List[Tuple[int, Action]] = []
            worse_actions: List[Tuple[int, Action]] = []
            current_suite = obs.declaration.suite if obs.declaration else TrumpSuite.XJ
            for action in obs.actions:
                assert isinstance(action, PlaceKittyAction)
                action: PlaceKittyAction
                
                rank = get_rank(action.card, current_suite, obs.dominant_rank)
                suite = get_suite(action.card, current_suite, obs.dominant_rank)
                
                # Best actions
                if suite != CardSuite.TRUMP and rank <= 11 and action.count == 1:
                    best_actions.append((rank, action))
                
                # Second best actions
                elif suite != CardSuite.TRUMP and (rank <= 8 and action.count == 2) or (rank <= 13 and action.count == 1):
                    second_best_actions.append((rank * 2, action))
                elif suite == CardSuite.TRUMP and rank <= 7:
                    worse_actions.append((rank, action))
            
            if best_actions:
                return min(best_actions, key=lambda x: x[0])[1]
            elif second_best_actions:
                return min(second_best_actions, key=lambda x: x[0])[1]
            else:
                return min(worse_actions, key=lambda x: x[0])[1]
        elif obs.stage == Stage.chaodi_stage:
            best_suite_action: Action = None
            nt_action: Action = None
            current_trump_count = obs.hand.count_suite(CardSuite.TRUMP, dominant_suite, obs.dominant_rank)
            best_chaodi_trump_count = 0
            for option in obs.actions:
                if isinstance(option, ChaodiAction):
                    if option.declaration.suite in ('XJ', 'DJ'):
                        nt_action = option
                    else:
                        new_trump_count = obs.hand.count_suite(CardSuite.TRUMP, option.declaration.suite, obs.dominant_rank)
                        best_chaodi_trump_count = max(best_chaodi_trump_count, new_trump_count)
            
            if best_suite_action and best_chaodi_trump_count >= 10 and best_chaodi_trump_count + 2 >= current_trump_count:
                return best_suite_action
            elif nt_action and current_trump_count < 12:
                return nt_action
            else:
                return DontChaodiAction()

        else:
            if obs.leads_current_round:
                optimal_actions = []
                second_best_actions = []
                other_actions = []
                for action in obs.actions:
                    action: LeadAction
                    d = action.move.cardset.decompose(dominant_suite, obs.dominant_rank)
                    
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
                    elif action.move.cardset.count_suite(CardSuite.TRUMP, dominant_suite, obs.dominant_rank) == 0:
                        min_rank = d[0].cardset.min_rank(dominant_suite, obs.dominant_rank)
                        if isinstance(d[0], MoveType.Tractor) or isinstance(d[0], MoveType.Pair) and min_rank >= 8 and isinstance(d[0], MoveType.Single) and (min_rank == 14 or obs.dominant_rank == 14 and min_rank == 13):
                            optimal_actions.append(action)
                        elif isinstance(d[0], MoveType.Pair) or d[0].cardset.min_rank(dominant_suite, obs.dominant_rank) > 10:
                            second_best_actions.append(action)
                        else:
                            other_actions.append(action)
                    else:
                        if action.move.cardset.total_points() == 0:
                            optimal_actions.append(action)
                        else:
                            other_actions.append(action)
                    
                    if optimal_actions:
                        return random.choice(optimal_actions)
                    elif second_best_actions:
                        return random.choice(second_best_actions)
                    else:
                        return other_actions[0]
                            
            else:
                current_round_points = 0
                dominant_suite = obs.declaration.suite if obs.declaration else TrumpSuite.XJ
                for h in obs.round_history[-1][1]:
                    current_round_points += h.total_points()
                
                good_action_choices = []
                other_action_choices = []
                lead_index = CardSet.round_winner(obs.round_history[-1][1], dominant_suite, obs.dominant_rank)
                teammate_is_biggest = len(obs.round_history[-1][1]) - lead_index == 2
                
                for action in obs.actions:
                    if not isinstance(action, FollowAction): continue

                    is_dominating = CardSet.round_winner(obs.round_history[-1][1] + [action.cardset], dominant_suite, obs.dominant_rank) == len(obs.round_history[-1][1])
                    if (current_round_points > 0 or len(obs.round_history[-1][1]) == 2) and not teammate_is_biggest:
                        if is_dominating:
                            good_action_choices.append(action)
                        else:
                            other_action_choices.append(action)
                    else:
                        good_action_choices.append(action)
                
                if good_action_choices:
                    return random.choice(good_action_choices)
                else:
                    return random.choice(other_action_choices)
                

