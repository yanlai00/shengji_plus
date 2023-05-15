from .Agent import SJAgent
from typing import List
import sys

sys.path.append('.')
from env.utils import ORDERING_INDEX, Stage, softmax
from env.Observation import Observation
from networks.Models import *

class InteractiveAgent(SJAgent):
    def act(self, obs: Observation, **kwargs):
        print(f'current hand ({obs.position.value}):', obs.hand)
        for i, a in enumerate(obs.actions):
            print(f'{i:>5}\t{a}')
        while True:
            idx = input("Enter action index: ")
            try:
                if obs.stage == Stage.main_stage:
                    return obs.actions[int(idx)], None, None
                else:
                    return obs.actions[int(idx)]
            except:
                continue