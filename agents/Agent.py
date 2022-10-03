
import logging
import random
import sys
from typing import List, Tuple

sys.path.append('.')
from env.Observation import Observation
from env.Actions import Action, DeclareAction, DontChaodiAction, DontDeclareAction, FollowAction, LeadAction, PlaceKittyAction
from env.utils import AbsolutePosition, Declaration, TrumpSuite
from env.CardSet import CardSet, MoveType

from env.utils import AbsolutePosition

class SJAgent:
    def __init__(self, name: str) -> None:
        self.name = name
    
    def act(self, obs: Observation):
        raise NotImplementedError
    
    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float]]):
        raise NotImplementedError

class RandomAgent(SJAgent):
    def act(self, obs: Observation):
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