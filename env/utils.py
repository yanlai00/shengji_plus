# Helper classes, functions and enums

import random
from enum import Enum
from typing import List, Union
import torch

class TrumpSuite(str, Enum):
    "All the possible suites for a trump declaration."
    CLUB = "♣"
    SPADE = "♠"
    HEART = "♥"
    DIAMOND = "♦"
    XJ = "XJ" # NT type 1
    DJ = "DJ" # NT type 2

    @property
    def is_NT(self):
        return self == 'XJ' or self == 'DJ'
    
    @property
    def tensor(self):
        rep = torch.zeros(6)
        idx = [self.DIAMOND, self.CLUB, self.HEART, self.SPADE, self.XJ, self.DJ].index(self)
        rep[idx] = 1
        return rep
    
    @classmethod
    def from_tensor(self, tensor: torch.Tensor) -> None:
        assert tensor.shape[0] == 6 and torch.sum(tensor == 1) == 1 and tensor.sum() == 1, "tensor must be one hot encoded"
        return [self.DIAMOND, self.CLUB, self.HEART, self.SPADE, self.XJ, self.DJ][tensor.argmax()]

class CardSuite(str, Enum):
    "All suites that a card can belong to in a game."
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
    
    @classmethod
    def random(self):
        return random.choice([AbsolutePosition.NORTH, AbsolutePosition.SOUTH, AbsolutePosition.EAST, AbsolutePosition.WEST])

class Declaration:
    "Contains information about the trump suite being declared."
    def __init__(self, suite: TrumpSuite, level: int, position: AbsolutePosition, relative_position: RelativePosition = None) -> None:
        self.suite = suite
        self.level = level
        self.absolute_position: AbsolutePosition = position
        self.relative_position = relative_position # Depends on the position of the player that observes this declaration

    def __repr__(self) -> str:
        return f"Declaration(player={self.absolute_position}, cards={self.suite.value} x{1 + int(self.level > 1)})"
    
    def relative_to(self, position: AbsolutePosition):
        return Declaration(self.suite, self.level, self.absolute_position, self.absolute_position.relative_to(position))

    @classmethod
    def chaodi_level(self, suite: TrumpSuite, level: int):
        if level > 1:
            if suite == TrumpSuite.DIAMOND:
                return 1
            elif suite == TrumpSuite.CLUB:
                return 2
            elif suite == TrumpSuite.HEART:
                return 3
            elif suite == TrumpSuite.SPADE:
                return 4
            elif suite == TrumpSuite.XJ:
                return 5
            else: # DJ
                return 6
        else:
            return 0

    
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

def get_suite(card: str, dominant_suite: TrumpSuite, dominant_rank: int):
    "Determines if the card is a trump card, and if not, determines which suite it is in."
    if card == 'XJ' or card == 'DJ':
        return CardSuite.TRUMP
    
    rank = NUMERIC_RANK[card[:-1]]
    suite = CardSuite(card[-1])
    
    if rank == dominant_rank or suite == dominant_suite:
        return CardSuite.TRUMP
    else:
        return suite

def get_rank(card: str, dominant_suite: TrumpSuite, dominant_rank: int):
    "Get the rank of a card within its suite."

    if card == 'DJ':
        return 18
    elif card == 'XJ':
        return 17
    else:
        suite = CardSuite(card[-1])
        rank = NUMERIC_RANK[card[:-1]]
        if rank == dominant_rank and (suite == dominant_suite or dominant_suite.is_NT):
            return 16
        elif rank == dominant_rank:
            return 15
        elif rank < dominant_rank:
            return rank + 1 # shift the rank of cards smaller than dominant rank by 1 to support tractors across the dominant rank
        else:
            return rank
