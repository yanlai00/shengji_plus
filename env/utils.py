# Helper classes, functions and enums

import random
from enum import Enum
import torch
import numpy as np

ORDERING = ['A♦', 'K♦', 'Q♦', 'J♦', '10♦', '9♦', '8♦', '7♦', '6♦', '5♦', '4♦', '3♦', '2♦', 'A♣', 'K♣', 'Q♣', 'J♣', '10♣', '9♣', '8♣', '7♣', '6♣', '5♣', '4♣', '3♣', '2♣', 'A♥', 'K♥', 'Q♥', 'J♥', '10♥', '9♥', '8♥', '7♥', '6♥', '5♥', '4♥', '3♥', '2♥', 'A♠', 'K♠', 'Q♠', 'J♠', '10♠', '9♠', '8♠', '7♠', '6♠', '5♠', '4♠', '3♠', '2♠', 'XJ', 'DJ']
ORDERING_INDEX = {k:i for i, k in enumerate(ORDERING)}

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

class CardSuit(str, Enum):
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
    
    @property
    def last_position(self):
        return {'N': AbsolutePosition.EAST, 'W': AbsolutePosition.NORTH, 'S': AbsolutePosition.WEST, 'E': AbsolutePosition.SOUTH}[self]

    @classmethod
    def random(self):
        return random.choice([AbsolutePosition.NORTH, AbsolutePosition.SOUTH, AbsolutePosition.EAST, AbsolutePosition.WEST])

class Stage(str, Enum):
    declare_stage = 'DECLARE'
    kitty_stage = 'KITTY'
    chaodi_stage = 'CHAODI'
    play_stage = 'PLAY'

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
    
    @property
    def tensor(self):
        "A tensor of shape (7,) representing the suite and multiplicity of the declaration."
        return torch.cat([self.suite.tensor, torch.tensor([int(self.level > 1)])])
    
    def get_card(self, dominant_rank: int):
        rank_symbol = LETTER_RANK[dominant_rank]
        count = 1 if self.level == 0 else 2
        if self.suite == TrumpSuite.XJ or self.suite == TrumpSuite.DJ:
            return self.suite.value, count
        else:
            return rank_symbol + self.suite.value, count


    @classmethod
    def chaodi_level(self, suite: TrumpSuite, level: int):
        if level >= 1:
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
        return CardSuit.TRUMP
    
    rank = NUMERIC_RANK[card[:-1]]
    suite = CardSuit(card[-1])
    
    if rank == dominant_rank or suite == dominant_suite:
        return CardSuit.TRUMP
    else:
        return suite

def get_rank(card: str, dominant_suite: TrumpSuite, dominant_rank: int):
    "Get the rank of a card within its suite."

    if card == 'DJ':
        return 18
    elif card == 'XJ':
        return 17
    else:
        suite = CardSuit(card[-1])
        rank = NUMERIC_RANK[card[:-1]]
        if rank == dominant_rank and (suite == dominant_suite or dominant_suite.is_NT):
            return 16
        elif rank == dominant_rank:
            return 15
        elif rank < dominant_rank:
            return rank + 1 # shift the rank of cards smaller than dominant rank by 1 to support tractors across the dominant rank
        else:
            return rank

def softmax(arr):
    return np.exp(arr) / np.sum(np.exp(arr))