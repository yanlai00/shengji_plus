
import sys

from env.Observation import Observation
sys.path.append('.')
from env.Actions import Action, DeclareAction, DontChaodiAction, DontDeclareAction, LeadAction, PlaceKittyAction
from env.utils import AbsolutePosition, Declaration, TrumpSuite
from env.CardSet import CardSet
from env.Game import Game

from typing import List
from env.utils import AbsolutePosition

class SJAgent:
    def __init__(self, position: AbsolutePosition) -> None:
        self.position = position
    
    def get_actions(self, observation: Observation):
        "Return a list of all possible actions the agent can take given the current observation. The player making the observation must choose an action to take."
        actions: List[Action] = []

        if not observation.draw_completed: # Stage 1: drawing cards phase
            for suite, level in observation.hand.trump_declaration_options(observation.dominant_rank).items():
                if observation.declaration is None or observation.declaration.level < level:
                    actions.append(DeclareAction(Declaration(suite, level, self.position)))
            actions.append(DontDeclareAction())
        elif observation.hand.size > 25: # Stage 2: choosing the kitty
            for card, count in observation.hand._cards.items():
                if count > 0: actions.append(PlaceKittyAction(card, count))
        elif observation.is_chaodi_turn:
            actions.append(DontChaodiAction())
            # TODO: chaodi
        elif observation.leads_current_round:
            # For training purpose, maybe fire turn off combos?
            for move in observation.hand.get_leading_moves(observation.declaration.suite, observation.dominant_rank, include_combos=True):
                actions.append(LeadAction(move))
        
        return actions