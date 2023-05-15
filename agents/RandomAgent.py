from .Agent import SJAgent, StageModule
from typing import List
import sys
import random
from env.CardSet import CardSet

sys.path.append('.')
from env.Actions import Action
from env.utils import ORDERING_INDEX, Stage, softmax
from env.Observation import Observation
from networks.Models import *

class RandomActorModule(StageModule):
    def act(self, obs: Observation, epsilon=None, training=True):
        if obs.stage != Stage.main_stage:
            return random.choice(obs.actions)
        
        others: List[Action] = []
        for a in obs.actions:
            others.append(a)

        return random.choice(others), None, None

class RandomAgent(SJAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        
        self.main_module = RandomActorModule()
