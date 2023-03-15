from collections import deque
import logging
import pickle
import random
import sys
from typing import Deque, List, Tuple, TypeVar, Generic
import numpy as np
from env.CardSet import CardSet, MoveType

from .Agent import SJAgent, StageModule

sys.path.append('.')
from env.Actions import Action, ChaodiAction, DeclareAction, DontChaodiAction, DontDeclareAction, FollowAction, LeadAction, AppendLeadAction, EndLeadAction, PlaceAllKittyAction, PlaceKittyAction
from env.utils import ORDERING_INDEX, Stage, softmax
from env.Observation import Observation
from networks.Models import *

class DQNAgent(SJAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.value_loss_history = []

    def optimizer_states(self):
        return {
            'value_optim_state': 'TODO',
        }