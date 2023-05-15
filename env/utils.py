# Helper classes, functions and enums

import random
from enum import Enum
import torch
import numpy as np

ORDERING = ['A♦', 'K♦', 'Q♦', 'J♦', '10♦', '9♦', '8♦', '7♦', '6♦', '5♦', '4♦', '3♦', '2♦', 'A♣', 'K♣', 'Q♣', 'J♣', '10♣', '9♣', '8♣', '7♣', '6♣', '5♣', '4♣', '3♣', '2♣', 'A♥', 'K♥', 'Q♥', 'J♥', '10♥', '9♥', '8♥', '7♥', '6♥', '5♥', '4♥', '3♥', '2♥', 'A♠', 'K♠', 'Q♠', 'J♠', '10♠', '9♠', '8♠', '7♠', '6♠', '5♠', '4♠', '3♠', '2♠']
ORDERING_INDEX = {k:i for i, k in enumerate(ORDERING)}

class TrumpSuit(str, Enum):
    "All the possible suits for a trump declaration."
    CLUB = "♣"
    SPADE = "♠"
    HEART = "♥"
    DIAMOND = "♦"
    NT = "NT"

    @property
    def is_NT(self):
        return self == 'NT'
    
    @property
    def tensor(self):
        rep = torch.zeros(6)
        idx = [self.DIAMOND, self.CLUB, self.HEART, self.SPADE, self.NT].index(self)
        rep[idx] = 1
        return rep
    
    @classmethod
    def from_tensor(self, tensor: torch.Tensor) -> None:
        assert tensor.shape[0] == 6 and torch.sum(tensor == 1) == 1 and tensor.sum() == 1, "tensor must be one hot encoded"
        return [self.DIAMOND, self.CLUB, self.HEART, self.SPADE, self.NT][tensor.argmax()]

class CardSuit(str, Enum):
    "All suits that a card can belong to in a game."
    CLUB = "♣"
    SPADE = "♠"
    HEART = "♥"
    DIAMOND = "♦"
    TRUMP = 'T'

class RelativePosition(str, Enum):
    LEFT = 'L'
    RIGHT = 'R'
    OPPOSITE = 'O'
    SELF = 'S'

class AbsolutePosition(str, Enum):
    NORTH = 'N'
    SOUTH = 'S'
    EAST = 'E'
    WEST = 'W'

    def relative_to(self, position: 'AbsolutePosition') -> RelativePosition:
        "Helper function that converts an absolute seat position to a relative seat position."
        return ['S', 'L', 'O', 'R'][(['N', 'W', 'S', 'E'].index(position) - ['N', 'W', 'S', 'E'].index(self)) % 4]

    @property
    def next_position(self):
        return {'N': AbsolutePosition.WEST, 'W': AbsolutePosition.SOUTH, 'S': AbsolutePosition.EAST, 'E': AbsolutePosition.NORTH}[self]
    
    @property
    def last_position(self):
        return {'N': AbsolutePosition.EAST, 'W': AbsolutePosition.NORTH, 'S': AbsolutePosition.WEST, 'E': AbsolutePosition.SOUTH}[self]

    @classmethod
    def random(self):
        return random.choice([AbsolutePosition.NORTH, AbsolutePosition.SOUTH, AbsolutePosition.EAST, AbsolutePosition.WEST])

class Stage(str, Enum):
    main_stage = 'PLAY'

class Declaration:
    "Contains information about the trump suit being declared."
    def __init__(self, suit: TrumpSuit, level: int, position: AbsolutePosition, relative_position: RelativePosition = None) -> None:
        self.suit = suit
        self.level = level
        self.absolute_position: AbsolutePosition = position
        self.relative_position = relative_position # Depends on the position of the player that observes this declaration

    def __repr__(self) -> str:
        return f"Declaration(player={self.absolute_position}, cards={self.suit.value} x{1 + int(self.level > 1)})"
    
    def relative_to(self, position: AbsolutePosition):
        return Declaration(self.suit, self.level, self.absolute_position, self.absolute_position.relative_to(position))
    
    @property
    def tensor(self):
        "A tensor of shape (7,) representing the suit and multiplicity of the declaration."
        return torch.cat([self.suit.tensor, torch.tensor([int(self.level > 1)])])
    
LETTER_RANK = {
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: '10',
    11: 'J',
    12: 'Q',
    13: 'K',
    14: 'A',
}

NUMERIC_RANK = {v:k for k,v in LETTER_RANK.items()}

def get_suit(card: str, dominant_suit: TrumpSuit):
    "Determines if the card is a trump card, and if not, determines which suit it is in."
    suit = CardSuit(card[-1])
    if suit == dominant_suit:
        return CardSuit.TRUMP
    else:
        return suit

def get_rank(card: str):
    "Get the rank of a card within its suit."
    rank = NUMERIC_RANK[card[:-1]]
    return rank

def softmax(arr):
    return np.exp(arr) / np.sum(np.exp(arr))