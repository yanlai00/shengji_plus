# Represent a set of cards in a Shengji game.
from random import shuffle
import random
from typing import Dict, List, Union
import torch
import itertools

from env.utils import LETTER_RANK, CardSuite, TrumpSuite, get_suite

class MoveType:
    @property
    def multiplier(self):
        return 2
    def __repr__(self) -> str:
        return "MoveType"
    
    def suite(self, dominant_suite: TrumpSuite, dominant_rank: int) -> CardSuite:
        "Returns the suite of the move"
        raise NotImplementedError

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
    
    def count_suite(self, suite: CardSuite, dominant_suite: TrumpSuite, dominant_rank: int):
        "Count the total number of cards the set has in the given CardSuite."
        total_count = 0
        for card, count in self._cards.items():
            if get_suite(card, dominant_suite, dominant_rank) == suite:
                total_count += count
        return total_count

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
    
    def card_list(self):
        "Return a list of cards contained in this CardSet."
        ordering: List[str] = []
        for card, count in self._cards.items():
            ordering.extend([card] * count)
        return ordering

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

    def get_leading_moves(self, dominant_suite: TrumpSuite, dominant_rank: int):
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
            moves.append(MoveType.Tractor('XJ', 'DJ', CardSet({'DJ': 2, 'XJ': 2})))
        if dominant_card is not None and self._cards[dominant_card] == 2:
            if self._cards['XJ'] == 2: # Small joker + dominant card
                moves.append(MoveType.Tractor(dominant_card, 'XJ', CardSet({'XJ': 2, dominant_card: 2})))
            for suite in filter(lambda s: s != dominant_suite, ['♣', '♠', '♦', '♥']):
                subdominant_card = LETTER_RANK[dominant_rank] + suite
                if self._cards[subdominant_card] == 2:
                    moves.append(MoveType.Tractor(subdominant_card, dominant_card, CardSet({dominant_card: 2, subdominant_card: 2})))
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
    
    def get_matching_moves(self, move: MoveType, dominant_suite: TrumpSuite, dominant_rank: int):
        move_suite = move.suite(dominant_suite, dominant_rank) # We need to first figure out the suite of the move the player is following.

        # We don't currently support combination moves (甩牌) because they increase the action space exponentially.
        def matches_for_simple_move(move: MoveType, hand: CardSet):
            matches: List[CardSet] = []
            same_suite_cards = {card:v for card,v in hand._cards.items() if v > 0 and get_suite(card, dominant_suite, dominant_rank) == move_suite}
            suite_count = hand.count_suite(move_suite, dominant_suite, dominant_rank)

            if isinstance(move, MoveType.Single):
                if len(same_suite_cards) > 0:
                    for card in same_suite_cards:
                        matches.append(CardSet({card: 1}))
                else:
                    for card, count in hand._cards.items():
                        if count > 0:
                            matches.append(CardSet({card: 1}))
            elif isinstance(move, MoveType.Pair):
                if suite_count > len(same_suite_cards): # If total count is greater than variety count, then there must be a duplicate
                    for card, count in same_suite_cards.items():
                        if count == 2:
                            matches.append(CardSet({card: 2}))
                elif suite_count >= 2:
                    for c1, c2 in itertools.combinations(same_suite_cards):
                        matches.append(CardSet({c1: 1, c2: 1}))
                elif suite_count == 1:
                    fixed_card, _ = same_suite_cards.popitem()
                    for card, count in hand._cards.items():
                        if count > 0 and card != fixed_card:
                            matches.append(CardSet({fixed_card: 1, card: 1}))
                else:
                    chosen_records = set()
                    for card1, card2 in itertools.combinations(hand.card_list(), 2):
                        if (card1, card2) in chosen_records or (card2, card1) in chosen_records: continue # skip duplicates
                        if card1 == card2:
                            matches.append(CardSet({card1: 2}))
                        else:
                            chosen_records.add((card1, card2))
                            matches.append(CardSet({card1: 1, card2: 1}))
            elif isinstance(move, MoveType.Tractor):
                if suite_count >= move.cardset.size:
                    # If there are enough cards in the suite, enumerate all possible tractors and check if the player has them
                    tractors = [t for t in self.get_leading_moves(dominant_suite, dominant_rank) if isinstance(t, MoveType.Tractor) and t.cardset.size == move.cardset.size and t.suite(dominant_suite, dominant_rank) == move_suite]
                    matches.extend([t.cardset for t in tractors])
                if not matches: # if we did not find a tractor in the same suite, the player needs to first play pairs
                    pairs = set([card for (card, count) in same_suite_cards.items() if count == 2])
                    if len(pairs) * 2 >= move.cardset.size: # If the player as at least as many pairs as the tractor size, choose any combination of the appropriate length
                        for pair_subset in itertools.combinations(pairs, move.cardset.size // 2):
                            matches.append(CardSet({p:2 for p in pair_subset}))
                
                if not matches:
                    # The player doesn't have enough pairs to add up to the length of the tractor, so they must play all those pairs, but in addition, they also need to play other cards of the same suite so that they add up to the tractor’s size.
                    must_choose_cards = {p:2 for p in pairs}
                    remaining_suite_choices = [c for c in same_suite_cards if c not in pairs]
                    if len(pairs) * 2 + len(remaining_suite_choices) >= move.cardset.size:
                        for remaining_suite_cards in itertools.combinations(remaining_suite_choices, move.cardset.size - len(pairs) * 2):
                            cardset = CardSet(must_choose_cards)
                            for card in remaining_suite_cards:
                                cardset.add_card(card)
                            matches.append(cardset)
                    else:
                        for suite_choice in remaining_suite_choices:
                            must_choose_cards[suite_choice] = 1
                
                if not matches: # The player doesn't have enough suite cards to match the tractor. So they must choose among the rest.
                    remaining_other_choices = [c for c in self.card_list() if c not in must_choose_cards]
                    records = set()
                    for remaining_other_cards in itertools.combinations(remaining_other_choices, move.cardset.size - len(pairs) * 2 - len(remaining_suite_choices)):
                        cardset = CardSet(must_choose_cards)
                        for card in remaining_other_cards:
                            cardset.add_card(card)
                        if str(cardset) in records: continue
                        records.add(str(cardset))
                        matches.append(cardset)

            return matches

        return matches_for_simple_move(move, self)
                

    def __str__(self) -> str:
        return str({k:v for k, v in self._cards.items() if v > 0})
    
    def __repr__(self) -> str:
        return f"CardSet({self})"

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
    "All possible valid moves."
    class Single(MoveType):
        def __init__(self, card: str) -> None:
            self.card = card
        def __repr__(self) -> str:
            return f"Single({self.card})"
        def suite(self, dominant_suite: TrumpSuite, dominant_rank: int) -> CardSuite:
            return get_suite(self.card, dominant_suite, dominant_rank)

    class Pair(MoveType):
        def __init__(self, card: str) -> None:
            self.card = card
        def __repr__(self) -> str:
            return f"Pair({self.card})"
        @property
        def multiplier(self):
            return 4
        def suite(self, dominant_suite: TrumpSuite, dominant_rank: int) -> CardSuite:
            return get_suite(self.card, dominant_suite, dominant_rank)
    
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
        def suite(self, dominant_suite: TrumpSuite, dominant_rank: int) -> CardSuite:
            return get_suite(self.low_card, dominant_suite, dominant_rank)
    
    class Combo(MoveType):
        "An active combo move. All elements in a combo are required to have the same CardSuite."
        def __init__(self, components: List[Union['MoveType.Single', 'MoveType.Pair', 'MoveType.Tractor']]) -> None:
            self.components = components
        def __repr__(self) -> str:
            return f"Combo({self.components})"
        @property
        def multiplier(self):
            return max([c.multiplier for c in self.components])
        def suite(self, dominant_suite: TrumpSuite, dominant_rank: int) -> CardSuite:
            return self.components[0].suite(dominant_suite, dominant_rank)