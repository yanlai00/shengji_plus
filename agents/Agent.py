
import logging
import random
import sys

from env.Observation import Observation
sys.path.append('.')
from env.Actions import Action, DeclareAction, DontChaodiAction, DontDeclareAction, FollowAction, LeadAction, PlaceKittyAction
from env.utils import AbsolutePosition, Declaration, TrumpSuite
from env.CardSet import CardSet
from env.Game import Game

from env.utils import AbsolutePosition

class SJAgent:
    def __init__(self, name: str) -> None:
        self.name = name
    
    def act(self, obs: Observation):
        raise NotImplementedError

class RandomAgent(SJAgent):
    def act(self, obs: Observation):
        return random.choice(obs.actions)