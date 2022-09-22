# Represent a set of cards in a Shengji game.
from random import shuffle
import random
from typing import Dict, List, Union
import torch

from env.utils import LETTER_RANK, TrumpSuite

class CardSet:
    def __init__(self, cardmap: Dict[str, int] = {}) -> None:
        self._cards = {
            "A♣":  0,
            "K♣":  0,
            "Q♣":  0,
            "J♣":  0,
            "10♣": 0,
            "9♣":  0,
            "8♣":  0,
            "7♣":  0,
            "6♣":  0,
            "5♣":  0,
            "4♣":  0,
            "3♣":  0,
            "2♣":  0,

            "A♠":  0,
            "K♠":  0,
            "Q♠":  0,
            "J♠":  0,
            "10♠": 0,
            "9♠":  0,
            "8♠":  0,
            "7♠":  0,
            "6♠":  0,
            "5♠":  0,
            "4♠":  0,
            "3♠":  0,
            "2♠":  0,

            "A♦":  0,
            "K♦":  0,
            "Q♦":  0,
            "J♦":  0,
            "10♦": 0,
            "9♦":  0,
            "8♦":  0,
            "7♦":  0,
            "6♦":  0,
            "5♦":  0,
            "4♦":  0,
            "3♦":  0,
            "2♦":  0,

            "A♥":  0,
            "K♥":  0,
            "Q♥":  0,
            "J♥":  0,
            "10♥": 0,
            "9♥":  0,
            "8♥":  0,
            "7♥":  0,
            "6♥":  0,
            "5♥":  0,
            "4♥":  0,
            "3♥":  0,
            "2♥":  0,

            "DJ":  0,  # color joker
            "XJ":  0   # b&w joker
        }

        for card, count in cardmap.items():
            self._cards[card] = count
    
    def __len__(self):
        return sum(self._cards.values())
    
    @property
    def size(self):
        return self.__len__()

    def add_card(self, card: str, count=1):
        assert card in self._cards, "Card must be a valid string representation"
        self._cards[card] += count
    
    def draw_N_cards(self, N):
        assert N <= self.size, "Cannot draw more cards than there are available"
        ordering = []
        for card, count in self._cards.items():
            ordering.extend([card] * count)
        
        random.shuffle(ordering)
        selected_cards = ordering[:N]
        for card in selected_cards:
            self.remove_card(card)
        
        return selected_cards
        
    def remove_card(self, card: str):
        assert self._cards[card] >= 1, "Cannot play the given card"
        self._cards[card] -= 1

    def to_tensor(self):
        return torch.zeros((1, 1)) # TODO
    
    def total_points(self):
        "Count the total number of points in the current set."
        points = 0
        for card, count in self._cards.items():
            if card.startswith('10') or card.startswith('K'):
                points += 10 * count
            elif card.startswith('5'):
                points += 5 * count
        return points
    
    def trump_declaration_options(self, rank: int):
        """Find all the possible trump suites that the user can declare, as well as their level. Higher level declarations override lower ones.
        
        Pair of same suite dominant rank cards: level 1
        Pair of b&w jokers: level 2
        Pair of color jokers: level 3
        """
        options: Dict[str, int] = {}
        if self._cards["DJ"] == 2:
            options["DJ"] = 3
        if self._cards["XJ"] == 2:
            options["XJ"] = 2
        
        letter_rank = LETTER_RANK[rank]
        for suite in [TrumpSuite.CLUB, TrumpSuite.DIAMOND, TrumpSuite.HEART, TrumpSuite.SPADE]:
            if self._cards[letter_rank + suite] >= 1:
                options[suite] = self._cards[letter_rank + suite]
        
        return options
    
    def get_count(self, suite: TrumpSuite, rank: int):
        if suite == TrumpSuite.DJ or suite == TrumpSuite.XJ:
            return self._cards['DJ'] + self._cards['XJ']
        return self._cards[LETTER_RANK[rank] + suite]

    def get_leading_moves(self, dominant_rank: int, dominant_suite: TrumpSuite):
        "Return all possible move actions the player can make if they lead the trick."
        moves: List[MoveType] = []

        # First add all singles and pairs
        for card, count in self._cards.items():
            if count >= 1:
                moves.append(MoveType.Single(card))
            if count == 2:
                moves.append(MoveType.Pair(card))
            
        # Add all tractors
        if not dominant_suite.is_NT:
            dominant_card = LETTER_RANK[dominant_rank] + dominant_suite
        else:
            dominant_card = None

        if self._cards['DJ'] == 2 and self._cards['XJ'] == 2: # Four jokers always form a tractor
            moves.append(MoveType.Tractor('XJ', 'DJ'))
        if dominant_card is not None and self._cards[dominant_card] == 2:
            if self._cards['XJ'] == 2: # Small joker + dominant card
                moves.append(MoveType.Tractor(dominant_card, 'XJ'))
            for suite in filter(lambda s: s != dominant_suite, ['♣', '♠', '♦', '♥']):
                subdominant_card = LETTER_RANK[dominant_rank] + suite
                if self._cards[subdominant_card] == 2:
                    moves.append(MoveType.Tractor(subdominant_card, dominant_card, CardSet({ dominant_card: 2, subdominant_card: 2 })))
        elif self._cards['XJ'] == 2:
            # In NT mode, any pair of rank cards can be paired with small jokers to form a tractor
            for suite in ['♣', '♠', '♦', '♥']:
                rank_card = LETTER_RANK[dominant_rank] + suite
                if self._cards[rank_card] == 2:
                    moves.append(MoveType.Tractor(rank_card, 'XJ', CardSet({ 'XJ': 2, rank_card: 2 })))
        
        # Other ordinary tractors
        ranks = list(range(2, 15)) # 2 to A
        ranks.remove(dominant_rank)
        for suite in ['♣', '♠', '♦', '♥']:
            for start_index in range(len(ranks) - 1):
                start_card = LETTER_RANK[ranks[start_index]] + suite # The starting point of a potential tractor
                if self._cards[start_card] < 2: continue
                tractor_set = CardSet({start_card: 2})
                end_index = start_index + 1
                while end_index < len(ranks) and self._cards[LETTER_RANK[ranks[end_index]] + suite] == 2:
                    end_card = LETTER_RANK[ranks[end_index]] + suite
                    tractor_set.add_card(end_card, count=2)
                    moves.append(MoveType.Tractor(start_card, end_card, tractor_set))
                    tractor_set = CardSet(tractor_set._cards) # Avoid mutation, need to duplicate
                    end_index += 1

        return moves

    def __str__(self) -> str:
        return str({k:v for k, v in self._cards.items() if v > 0})

    @classmethod
    def new_deck(self):
        "Initialize a new CardSet object with two decks of cards and a random order."
        deck = CardSet()
        ordering = []
        for card in deck._cards:
            deck._cards[card] = 2
            ordering.extend([card] * 2)
        
        shuffle(ordering)
        return deck, ordering

        

class MoveType:
    @property
    def multiplier(self):
        return 2
    def __repr__(self) -> str:
        return "MoveType"

class MoveType:
    "All possible valid moves."
    class Single(MoveType):
        def __init__(self, card: str) -> None:
            self.card = card
        def __repr__(self) -> str:
            return f"Single({self.card})"

    class Pair(MoveType):
        def __init__(self, card: str) -> None:
            self.card = card
        def __repr__(self) -> str:
            return f"Pair({self.card})"
        @property
        def multiplier(self):
            return 4
    
    class Tractor(MoveType):
        def __init__(self, low_card: str, high_card: str, cardset: CardSet) -> None:
            self.low_card = low_card
            self.high_card = high_card
            self.cardset = cardset
        def __repr__(self) -> str:
            return f"Tractor({self.cardset})"
        @property
        def multiplier(self):
            return 16 if self.length == 2 else 64
    
    class Combo(MoveType):
        def __init__(self, components: List[Union['MoveType.Single', 'MoveType.Pair', 'MoveType.Tractor']]) -> None:
            self.components = components
        def __repr__(self) -> str:
            return f"Combo({self.components})"
        @property
        def multiplier(self):
            return max([c.multiplier for c in self.components])