# Defines all potential actions that a player can take during a game.
from .CardSet import CardSet, Move
from .utils import LETTER_RANK, ORDERING, ORDERING_INDEX, CardSuit, TrumpSuit, get_rank, get_suit
import torch

class Action:
    "The base class of all actions. Should never be used directly."

    def __repr__(self) -> str:
        return "Action()"
    
    @property
    def tensor(self) -> torch.Tensor:
        raise NotImplementedError

class LeadAction(Action):
    def __init__(self, move: Move) -> None:
        self.move = move
    def __repr__(self) -> str:
        return f"LeadAction({self.move})"
    @property
    def cardset(self):
        return self.move.cardset
    @property
    def tensor(self) -> torch.Tensor:
        "Shape: (TOTAL_CARDS,)"
        return self.move.cardset.tensor
    
    def dynamic_tensor(self, dominant_suit: TrumpSuit):
        return self.move.cardset.get_dynamic_tensor(dominant_suit)

class FollowAction(Action):
    "Follow the pattern of a previous player. Everyone except the first player in a trick will need to follow the pattern."
    def __init__(self, cardset: CardSet) -> None:
        self.cardset = cardset
    def __repr__(self) -> str:
        return f"FollowAction({self.cardset})"
    @property
    def tensor(self) -> torch.Tensor:
        "Shape: (TOTAL_CARDS,)"
        return self.cardset.tensor
    
    def dynamic_tensor(self, dominant_suit: TrumpSuit):
        return self.cardset.get_dynamic_tensor(dominant_suit)
