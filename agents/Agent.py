
import logging
import random
import sys
from typing import List, Tuple
import numpy as np
import pickle
import torch
from torch import nn

sys.path.append('.')
from env.Observation import Observation
from env.Actions import Action
from env.utils import Stage
from env.CardSet import CardSet

from env.utils import AbsolutePosition

class StageModule:
    def act(self, obs: Observation, epsilon=None, training=True):
        return NotImplementedError()

    def load_model(self, model: nn.Module):
        raise NotImplementedError()

    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float]]):
        raise NotImplementedError()

class SJAgent:
    def __init__(self, name: str) -> None:
        self.name = name

        self.main_module: StageModule = None
    
    def act(self, obs: Observation, epsilon=None, training=True):
        assert self.main_module is not None, "At least one required model is not loaded"
        if obs.stage == Stage.main_stage:
            return self.main_module.act(obs, epsilon, training)
        else:
            raise NotImplementedError()
    
    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float]], stage: Stage):
        if stage == Stage.main_stage:
            self.main_module.learn_from_samples(samples)
        else:
            raise NotImplementedError()

    def optimizer_states(self):
        return {}

    def load_optimizer_states(self, state):
        pass
    
    # Try to load models from disk. Return whether the models were loaded successfully.
    def load_models_from_disk(self) -> bool:
        raise NotImplementedError()

    def save_models_to_disk(self):
        raise NotImplementedError()

