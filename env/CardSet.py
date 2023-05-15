# Represent a set of cards in a Shengji game.
import json
from random import shuffle
import random
from typing import Dict, List, Set, Tuple, Union
import torch
import itertools

from env.utils import LETTER_RANK, ORDERING, CardSuit, TrumpSuit, get_rank, get_suit
from game_consts import TOTAL_CARDS

class Move():
    def __init__(self, card: str) -> None:
        self.card = card

    def __repr__(self) -> str:
        return f"Move({self.card})"
    
    @property
    def cardset(self):
        return CardSet({self.card: 1})
    
    def suit(self, dominant_suit: TrumpSuit) -> CardSuit:
        return get_suit(self.card, dominant_suit)

class CardSet:
    def __init__(self, cardmap: Dict[str, int] = {}) -> None:
        self._cards: Dict[str, int] = {card : 0 for card in ORDERING}

        for card, count in cardmap.items():
            self._cards[card] = count
    
    def __len__(self):
        return sum(self._cards.values())

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, CardSet):
            return False
        return self._cards == __o._cards
    
    def __hash__(self) -> int:
        return hash(str(self))
    
    def contains(self, other_cardset: 'CardSet'):
        for card in other_cardset._cards:
            if self._cards[card] < other_cardset._cards[card]:
                return False
        return True
    
    def has_card(self, card: str):
        return self._cards[card] > 0
    
    def filter_by_suit(self, suit: CardSuit, dominant_suit: TrumpSuit):
        subset = CardSet()
        for card, count in self._cards.items():
            if count > 0 and get_suit(card, dominant_suit) == suit:
                subset.add_card(card, count)
        return subset

    @property
    def size(self):
        return self.__len__()
    
    def count_suit(self, suit: CardSuit, dominant_suit: TrumpSuit):
        "Count the total number of cards the set has in the given CardSuit."
        total_count = 0
        for card, count in self._cards.items():
            if get_suit(card, dominant_suit) == suit:
                total_count += count
        return total_count

    def add_card(self, card: str, count=1):
        assert card in self._cards, "Card must be a valid string representation"
        self._cards[card] += count
    
    def add_cardset(self, cardset: 'CardSet'):
        for card in cardset.card_list():
            self.add_card(card) 
        
    def remove_card(self, card: str, count=1, strict=True):
        if strict:
            assert self._cards[card] >= count, "Cannot play the given card"
        if self._cards[card] >= count:
            self._cards[card] -= count
        else:
            self._cards[card] = 0
    
    def remove_cardset(self, cardset: 'CardSet', strict=True):
        for card in cardset.card_list():
            self.remove_card(card, strict=strict)
    
    def card_list(self):
        "Return a list of cards contained in this CardSet."
        ordering: List[str] = []
        for card, count in self._cards.items():
            ordering.extend([card] * count)
        return ordering
    
    def get_count(self, suit: TrumpSuit, rank: int):
        if suit == TrumpSuit.NT:
            return 0
        return self._cards[LETTER_RANK[rank] + suit]

    def get_leading_moves(self):
        "Return all possible move actions the player can make if they lead the trick."
        moves: List[Move] = []

        # First add all singles and pairs
        for card, count in self._cards.items():
            moves.append(Move(card))

        return moves

    def get_matching_moves(self, target: Move, dominant_suit: TrumpSuit,):
        target_suit = target.suit(dominant_suit) # We need to first figure out the suit of the move the player is following.
        
        matches: Set[CardSet] = set()
        same_suit_cards = {card:v for card,v in self._cards.items() if v > 0 and get_suit(card, dominant_suit) == target_suit}
    
        if len(same_suit_cards) > 0:
            for card in same_suit_cards:
                matches.add(CardSet({card: 1}))
        else:
            for card, count in self._cards.items():
                if count > 0:
                    matches.add(CardSet({card: 1}))
        
        # Sorting makes the output order deterministic
        return sorted(matches, key=lambda c: c.__str__())

    def is_bigger_than(self, move: Move, dominant_suit: TrumpSuit,):
        "Determine if this cardset can beat a given move. If so, return the decomposition. If not, return None."
        # assert self.size == move.cardset.size, "CardSets must be the same size to compare"

        self_components = self.get_leading_moves()
        first_component = max(self_components, key=lambda c: c.cardset.size)
        target_suit = move.suit(dominant_suit)
        self_suit = first_component.suit(dominant_suit) 

        if type(first_component) != type(move): 
            return None
        self_rank = get_rank(first_component.card)
        target_rank = get_rank(move.card)
        
        if self_suit == target_suit and self_rank > target_rank or target_suit != CardSuit.TRUMP and self_suit == CardSuit.TRUMP:
            return [first_component]

    def min_rank(self):
        return min([get_rank(x) for x in self.card_list()])

    def __str__(self) -> str:
        return str({k:v for k, v in self._cards.items() if v > 0})
    
    def __repr__(self) -> str:
        return f"CardSet({self})"
    
    def copy(self):
        return CardSet(self._cards)
    
    def count_iterator(self):
        for (card, count) in self._cards.items():
            if count > 0:
                yield (card, count)

    @classmethod
    def new_deck(self):
        "Initialize a new CardSet object with two decks of cards and a random order."
        deck = CardSet()
        ordering = []
        for card in deck._cards:
            deck._cards[card] = 1
            ordering.extend([card] * 1)
        
        shuffle(ordering)
        return deck, ordering

    @classmethod
    def round_winner(self, cardsets: List['CardSet'], dominant_suit: TrumpSuit):
        "Determine the position of the largest move."

        leading_move = Move(cardsets[0])
        follow_cardsets = cardsets[1:]
        if len(follow_cardsets) == 0:
            return 0

        configs = [move.is_bigger_than(leading_move, dominant_suit) for move in follow_cardsets]
        
        def key_fn(move: Move):
            "Returns the maximum rank of the largest component of a move"
            return (move.cardset.size, max([get_rank(card) for card in move.cardset.card_list()]))

        ranks = [1] + [0] * len(follow_cardsets)
        for i, config in enumerate(configs, start=1):
            if config:
                best_component: Move = max(config, key=key_fn)
                # Add 100 to trump components so that they are definitely bigger than non-trump components.
                ranks[i] = key_fn(best_component)[1] + (100 if best_component.suit(dominant_suit) == CardSuit.TRUMP else 0)
        
        return max(range(len(cardsets)), key=lambda i: ranks[i]) 

    @property
    def tensor(self):
        "Return a fixed size binary tensor of shape (TOTAL_CARDS,) representing the cardset."
        rep = torch.zeros(TOTAL_CARDS)
        for i, card in enumerate(ORDERING):
            if self._cards[card] >= 1:
                rep[i] = 1
        return rep
    
    @classmethod
    def from_tensor(self, t: torch.Tensor):
        cardset = CardSet()
        for i, card in enumerate(ORDERING):
            cardset.add_card(card, count=int(t[i]))
        return cardset

    @classmethod
    def create_deck_from_hands(self, hands: List[List[str]]):
        assert len(hands) == 4, 'There must be 4 provided hands'
        deck = []
        for card_list in hands:
            random.shuffle(card_list)
        for h1, h2, h3, h4 in zip(*hands):
            deck.extend([h1, h2, h3, h4])
        
        return deck
    
    def get_dynamic_tensor(self, trump_suit: TrumpSuit):
        rep = torch.zeros(TOTAL_CARDS)
        
        index = 0
        # in the order: diamonds, clubs, hearts, spades, ignoring trump cards
        for suit in [TrumpSuit.DIAMOND, TrumpSuit.CLUB, TrumpSuit.HEART, TrumpSuit.SPADE]:
            if not trump_suit.is_NT and suit == trump_suit: 
                continue
            for numeric_rank, letter_rank in LETTER_RANK.items():
                card = letter_rank + suit
                if self._cards[card] >= 1:
                    rep[index] = 1
                index += 1
        
        # Trump suit
        if not trump_suit.is_NT:
            for numeric_rank, letter_rank in LETTER_RANK.items():
                card = letter_rank + trump_suit
                rep[index:index + self._cards[card]] = 1
                index += 1

        return rep

    @classmethod
    def from_dynamic_tensor(self, rep: torch.Tensor, trump_suit: TrumpSuit):
        cardset = CardSet()

        index = 0
        for suit in [TrumpSuit.DIAMOND, TrumpSuit.CLUB, TrumpSuit.HEART, TrumpSuit.SPADE]:
            if not trump_suit.is_NT and suit == trump_suit: continue
            for numeric_rank, letter_rank in LETTER_RANK.items():
                card = letter_rank + suit
                cardset.add_card(card, count=rep[index])
                index += 1
        
        # Trump suit
        if not trump_suit.is_NT:
            for numeric_rank, letter_rank in LETTER_RANK.items():
                card = letter_rank + trump_suit
                cardset.add_card(card, count=rep[index])
                index += 1

        return cardset

    @classmethod
    def get_tutorial_deck(self):
        """
        A tutorial deck demonstrates the most extreme cases of the game. In this example, each player owns almost an entire suit.
        """

        # Each player starts with an entire suit
        hands = [
            CardSet({
                '2♠': 1, '3♠': 1, '4♠': 1, '5♠': 1, '6♠': 1, '7♠': 1, '8♠': 1, '9♠': 1, '10♠': 1, 'J♠': 1, 'Q♠': 1, 'K♠': 1, 'A♠': 1,
            }),
            CardSet({
                '2♦': 1, '3♦': 1, '4♦': 1, '5♦': 1, '6♦': 1, '7♦': 1, '8♦': 1, '9♦': 1, '10♦': 1, 'J♦': 1, 'Q♦': 1, 'K♦': 1, 'A♦': 1,
            }),
            CardSet({
                '2♣': 1, '3♣': 1, '4♣': 1, '5♣': 1, '6♣': 1, '7♣': 1, '8♣': 1, '9♣': 1, '10♣': 1, 'J♣': 1, 'Q♣': 1, 'K♣': 1, 'A♣': 1,
            }),
            CardSet({
                '2♥': 1, '3♥': 1, '4♥': 1, '5♥': 1, '6♥': 1, '7♥': 1, '8♥': 1, '9♥': 1, '10♥': 1, 'J♥': 1, 'Q♥': 1, 'K♥': 1, 'A♥': 1,
            })
        ]
        random.shuffle(hands)

        return self.create_deck_from_hands([hand.card_list() for hand in hands])
        

