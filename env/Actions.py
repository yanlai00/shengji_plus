# Defines all potential actions that a player can take during a game.
from .CardSet import CardSet, MoveType
from .utils import LETTER_RANK, Declaration, TrumpSuite, encode_single_card
import torch

class Action:
    "The base class of all actions. Should never be used directly."

    def __repr__(self) -> str:
        return "Action()"
    
    @property
    def tensor(self) -> torch.Tensor:
        raise NotImplementedError

class DeclareAction(Action):
    "Reveal one or more cards in the dominant rank (or a pair of identical jokers) to declare or override the trump suite."
    def __init__(self, declaration: Declaration) -> None:
        self.declaration = declaration
    def __repr__(self) -> str:
        return f"DeclareAction({self.declaration.suite}, lv={self.declaration.level})"
    @property
    def tensor(self) -> torch.Tensor:
        "Shape: (9,)"
        level_vector = torch.zeros(3)
        for i in range(self.declaration.level):
            level_vector[i] = 1
        return torch.cat([self.declaration.suite.tensor, level_vector])


class DontDeclareAction(Action):
    "Choose not to reveal any dominant rank card during the draw phase of the game."

    def __repr__(self) -> str:
        return "DontDeclareAction()"
    @property
    def tensor(self) -> torch.Tensor:
        return torch.zeros(9)

# Usually all possible actions are explicitly computed when a player is about to take an action. But there are (33 -> 8) possible ways for the dealer to select the kitty, so we decompose this stage into 8 actions. Each step the dealer choose one card to discard. He does this 8 times.
class PlaceKittyAction(Action):
    "Discards a card and places it in the kitty (dealer or chaodi player)."
    def __init__(self, card: str, count=1) -> None:
        self.card = card
        self.count = count
    def __repr__(self) -> str:
        if self.count == 1:
            return f"Discard({self.card})"
        else:
            return f"Discard({self.card}, count={self.count})"
    @property
    def tensor(self) -> torch.Tensor:
        "Shape: (54,)"
        return encode_single_card(self.card)
        

class ChaodiAction(Action):
    "Changes trump suite and swap cards with the kitty."
    def __init__(self, declaration: Declaration) -> None:
        self.declaration = declaration
    def __repr__(self) -> str:
        return f"ChaodAction({self.declaration})"
    @property
    def tensor(self) -> torch.Tensor:
        "Shape: (6,)"
        return self.declaration.suite.tensor

class DontChaodiAction(Action):
    "Skip chaodi."
    def __repr__(self) -> str:
        return f"DontChaodiAction()"
    @property
    def tensor(self) -> torch.Tensor:
        "Shape: (6,)"
        return torch.zeros(6)

class LeadAction(Action):
    "Start a new round by placing a single, pair, or tractor. For now, we disable combinations."
    def __init__(self, move: MoveType) -> None:
        self.move = move
    def __repr__(self) -> str:
        return f"LeadAction({self.move})"
    @property
    def tensor(self) -> torch.Tensor:
        "Shape: (54,2)"
        return self.move.cardset.tensor

class FollowAction(Action):
    "Follow the pattern of a previous player. Everyone except the first player in a trick will need to follow the pattern."
    def __init__(self, cardset: CardSet) -> None:
        self.cardset = cardset
    def __repr__(self) -> str:
        return f"FollowAction({self.cardset})"
    @property
    def tensor(self) -> torch.Tensor:
        "Shape: (54,2)"
        return self.cardset.tensor
