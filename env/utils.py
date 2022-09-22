# Helper classes, functions and enums

import random
from enum import Enum
from typing import List, Union

class TrumpSuite(str, Enum):
    CLUB = "♣"
    SPADE = "♠"
    HEART = "♥"
    DIAMOND = "♦"
    XJ = "XJ" # NT type 1
    DJ = "DJ" # NT type 2

    @property
    def is_NT(self):
        return self == 'XJ' or self == 'DJ'

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
        return ['S', 'L', 'O', 'R'][(['N', 'W', 'S', 'E'].index(self) - ['N', 'W', 'S', 'E'].index(position)) % 4]

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
        return f"Declaration(position={self.absolute_position}, {self.suite}, {self.level})"
    
    def relative_to(self, position: AbsolutePosition):
        return Declaration(self.suite, self.level, self.absolute_position, self.absolute_position.relative_to(position))

    
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