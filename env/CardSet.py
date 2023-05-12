# Represent a set of cards in a Shengji game.
import json
from random import shuffle
import random
from typing import Dict, List, Set, Tuple, Union
import torch
import itertools

from env.utils import LETTER_RANK, ORDERING, CardSuit, TrumpSuit, get_rank, get_suit

class MoveType:
    pass

class MoveType:
    @property
    def multiplier(self):
        return 2
    def __repr__(self) -> str:
        return "MoveType"
    @property
    def cardset(self) -> 'CardSet':
        return CardSet()
    def suit(self, dominant_suit: TrumpSuit, dominant_rank: int) -> CardSuit:
        "Returns the suit of the move"
        raise NotImplementedError
        
    "All possible valid moves."
    class Single(MoveType):
        def __init__(self, card: str) -> None:
            self.card = card
        def __repr__(self) -> str:
            return f"Single({self.card})"
        @property
        def cardset(self):
            return CardSet({self.card: 1})
        @property
        def multiplier(self):
            return 2
        def suit(self, dominant_suit: TrumpSuit, dominant_rank: int) -> CardSuit:
            return get_suit(self.card, dominant_suit, dominant_rank)

    class Pair(MoveType):
        def __init__(self, card: str) -> None:
            self.card = card
        def __repr__(self) -> str:
            return f"Pair({self.card})"
        @property
        def multiplier(self):
            return 4
        @property
        def cardset(self):
            return CardSet({self.card: 2})
        def suit(self, dominant_suit: TrumpSuit, dominant_rank: int) -> CardSuit:
            return get_suit(self.card, dominant_suit, dominant_rank)
    
    class Tractor(MoveType):
        def __init__(self, cardset: "CardSet") -> None:
            assert cardset.size >= 4, "Tractors must have at least 4 cards"
            self._cardset = cardset
        def __repr__(self) -> str:
            return f"Tractor({self._cardset})"
        @property
        def multiplier(self):
            return 16 if self._cardset.size == 4 else 64
        @property
        def cardset(self):
            return self._cardset
        def suit(self, dominant_suit: TrumpSuit, dominant_rank: int) -> CardSuit:
            return get_suit(self._cardset.card_list()[0], dominant_suit, dominant_rank)
    
    class Combo(MoveType):
        "An active combo move. All elements in a combo are required to have the same CardSuit."
        def __init__(self, cardset: 'CardSet') -> None:
            self._cardset = cardset
        def __repr__(self) -> str:
            return f"Combo({self._cardset})"
        @property
        def multiplier(self):
            print("Do not call .multiplier of combo directly! Use .get_multiplier(...) instead.")
            return 2
        def get_components(self, dominant_suit: TrumpSuit, dominant_rank: int):
            return self._cardset.decompose(dominant_suit, dominant_rank)
        def get_multiplier(self, dominant_suit: TrumpSuit, dominant_rank: int):
            return max([c.multiplier for c in self.get_components(dominant_suit, dominant_rank)])
        @property
        def cardset(self):
            return self._cardset
        def suit(self, dominant_suit: TrumpSuit, dominant_rank: int) -> CardSuit:
            return self.get_components(dominant_suit, dominant_rank)[0].suit(dominant_suit, dominant_rank)

    @classmethod
    def from_cardset(self, cardset: 'CardSet', dominant_suit: TrumpSuit, dominant_rank: int):
        components = cardset.decompose(dominant_suit, dominant_rank)
        if len(components) == 1:
            return components[0]
        else:
            return MoveType.Combo(cardset)

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
    
    def filter_by_suit(self, suit: CardSuit, dominant_suit: TrumpSuit, dominant_rank: int):
        subset = CardSet()
        for card, count in self._cards.items():
            if count > 0 and get_suit(card, dominant_suit, dominant_rank) == suit:
                subset.add_card(card, count)
        return subset

    @property
    def size(self):
        return self.__len__()
    
    def count_suit(self, suit: CardSuit, dominant_suit: TrumpSuit, dominant_rank: int):
        "Count the total number of cards the set has in the given CardSuit."
        total_count = 0
        for card, count in self._cards.items():
            if get_suit(card, dominant_suit, dominant_rank) == suit:
                total_count += count
        return total_count

    def add_card(self, card: str, count=1):
        assert card in self._cards, "Card must be a valid string representation"
        self._cards[card] += count
    
    def add_cardset(self, cardset: 'CardSet'):
        for card in cardset.card_list():
            self.add_card(card) 
    
    def draw_N_cards(self, N: int):
        assert N <= self.size, "Cannot draw more cards than there are available"
        ordering: List[str] = []
        for card, count in self._cards.items():
            ordering.extend([card] * count)
        
        random.shuffle(ordering)
        selected_cards = ordering[:N]
        for card in selected_cards:
            self.remove_card(card)
        
        return selected_cards
        
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
        """Find all the possible trump suits that the user can declare, as well as their level. Higher level declarations override lower ones.
        
        Pair of same suit dominant rank cards: level 1
        Pair of b&w jokers: level 2
        Pair of color jokers: level 3
        """
        options: Dict[TrumpSuit, int] = {}
        if self._cards["DJ"] == 2:
            options[TrumpSuit.DJ] = 3
        if self._cards["XJ"] == 2:
            options[TrumpSuit.XJ] = 2
        
        letter_rank = LETTER_RANK[rank]
        for suit in [TrumpSuit.CLUB, TrumpSuit.DIAMOND, TrumpSuit.HEART, TrumpSuit.SPADE]:
            if self._cards[letter_rank + suit] > 0:
                options[suit] = self._cards[letter_rank + suit] - 1 # Pair has level 1, single has level 0
        return options
    
    def get_count(self, suit: TrumpSuit, rank: int):
        if suit == TrumpSuit.DJ or suit == TrumpSuit.XJ:
            return self._cards['DJ'] + self._cards['XJ']
        return self._cards[LETTER_RANK[rank] + suit]

    def get_leading_moves(self, dominant_suit: TrumpSuit, dominant_rank: int, include_combos = False):
        "Return all possible move actions the player can make if they lead the trick."
        moves: List[MoveType] = []
        suit_list = [CardSuit.CLUB, CardSuit.DIAMOND, CardSuit.HEART, CardSuit.SPADE, CardSuit.TRUMP]
        
        if include_combos:
            cards_by_suit = {suit: CardSet() for suit in suit_list}
            for card in self.card_list():
                cards_by_suit[get_suit(card, dominant_suit, dominant_rank)].add_card(card)
            for suit, suit_cardset in cards_by_suit.items():
                if suit == CardSuit.TRUMP: continue
                records = set()
                for size in range(1, suit_cardset.size + 1):
                    for combo in itertools.combinations(suit_cardset.card_list(), size):
                        combo_cardset = CardSet()
                        for move in combo:
                            combo_cardset.add_card(move)
                        components = combo_cardset.decompose(dominant_suit, dominant_rank)
                        if combo_cardset not in records:
                            records.add(combo_cardset)
                            moves.append(MoveType.Combo(combo_cardset) if len(components) > 1 else components[0])
            if moves:
                return moves

        pair_cards: List[str] = []
        # First add all singles and pairs
        for card, count in self._cards.items():
            if count >= 1:
                moves.append(MoveType.Single(card))
            if count == 2:
                moves.append(MoveType.Pair(card))
                pair_cards.append(card)
            
        # Add all tractors
        if not dominant_suit.is_NT:
            suit_list.remove(dominant_suit)
        
        pairs_by_suit = {suit: [] for suit in suit_list}
        for card in pair_cards:
            pairs_by_suit[get_suit(card, dominant_suit, dominant_rank)].append(card)

        for suit, cards in pairs_by_suit.items():
            card_ranks: List[Tuple[int, str]] = []
            for card in cards:
                rank = get_rank(card, dominant_suit, dominant_rank)
                card_ranks.append((rank, card))
            
            grouped_ranks = [(r, list(cards)) for r, cards in itertools.groupby(sorted(card_ranks), key=lambda x: x[0])]

            for start_index in range(len(grouped_ranks) - 1):
                end_index = start_index + 1
                tractor_cards = [grouped_ranks[start_index][1]]
                while end_index < len(grouped_ranks) and end_index - start_index == grouped_ranks[end_index][0] - grouped_ranks[start_index][0]:
                    tractor_cards.append(grouped_ranks[end_index][1])
                    for tractor in itertools.product(*tractor_cards):
                        combo_cardset = CardSet()
                        for _, card in tractor: combo_cardset.add_card(card, count=2)
                        moves.append(MoveType.Tractor(combo_cardset))
                    end_index += 1

        return moves
    
    def decompose(self, dominant_suit: TrumpSuit, dominant_rank: int) -> List[Union[MoveType.Single, MoveType.Pair, MoveType.Tractor]]:
        "Decompose a leading move into components using a greedy approach."
        remaining_cards = self.copy()
        components = []
        while remaining_cards.size > 0:
            largest = max(remaining_cards.get_leading_moves(dominant_suit, dominant_rank), key=lambda move: move.cardset.size)
            remaining_cards.remove_cardset(largest.cardset)
            components.append(largest)
        
        return components

    def get_matching_moves(self, target: MoveType, dominant_suit: TrumpSuit, dominant_rank: int):
        target_suit = target.suit(dominant_suit, dominant_rank) # We need to first figure out the suit of the move the player is following.
        
        # Convenient helper function that returns a list of longest tractors in a CardSet
        def get_longest_tractors(cardset: CardSet) -> Tuple[int, List[MoveType.Tractor]]:
            tractors = sorted([t for t in cardset.get_leading_moves(dominant_suit, dominant_rank) if isinstance(t, MoveType.Tractor) and t.suit(dominant_suit, dominant_rank) == target_suit], key=lambda t: t.cardset.size, reverse=True)
            
            if not tractors: return 0, []

            longest_tractors = []
            for t in tractors:
                if t.cardset.size == tractors[0].cardset.size:
                    longest_tractors.append(t)
            return tractors[0].cardset.size, longest_tractors

        def matches_for_simple_move(target: Union[MoveType.Single, MoveType.Pair, MoveType.Tractor], hand: CardSet):
            matches: Set[CardSet] = set()
            same_suit_cards = {card:v for card,v in hand._cards.items() if v > 0 and get_suit(card, dominant_suit, dominant_rank) == target_suit}
            suit_count = hand.count_suit(target_suit, dominant_suit, dominant_rank)

            if isinstance(target, MoveType.Single):
                if len(same_suit_cards) > 0:
                    for card in same_suit_cards:
                        matches.add(CardSet({card: 1}))
                else:
                    for card, count in hand._cards.items():
                        if count > 0:
                            matches.add(CardSet({card: 1}))
            elif isinstance(target, MoveType.Pair):
                if suit_count > len(same_suit_cards): # If total count is greater than variety count, then there must be a pair
                    for card, count in same_suit_cards.items():
                        if count == 2:
                            matches.add(CardSet({card: 2}))
                elif suit_count >= 2:
                    for c1, c2 in itertools.combinations(same_suit_cards, 2):
                        matches.add(CardSet({c1: 1, c2: 1}))
                elif suit_count == 1:
                    fixed_card, _ = same_suit_cards.popitem()
                    for card, count in hand._cards.items():
                        if count > 0 and card != fixed_card:
                            matches.add(CardSet({fixed_card: 1, card: 1}))
                else:
                    chosen_records = set()
                    for card1, card2 in itertools.combinations(hand.card_list(), 2):
                        if (card1, card2) in chosen_records or (card2, card1) in chosen_records: continue # skip duplicates
                        if card1 == card2:
                            matches.add(CardSet({card1: 2}))
                        else:
                            chosen_records.add((card1, card2))
                            matches.add(CardSet({card1: 1, card2: 1}))
            elif isinstance(target, MoveType.Tractor):
                if suit_count >= target.cardset.size:
                    # If there are enough cards in the suit, enumerate all possible tractors and check if the player has them
                    
                    exact_tractors = [t.cardset for t in hand.get_leading_moves(dominant_suit, dominant_rank) if isinstance(t, MoveType.Tractor) and t.suit(dominant_suit, dominant_rank) == target_suit and t.cardset.size == target.cardset.size]
                    matches.update(exact_tractors)
                    
                    if not exact_tractors:                        
                        s, tractor_list = get_longest_tractors(CardSet(same_suit_cards))
                        if s > 0:
                            # Find all non-overlapping combinations of tractors
                            non_overlapping_tractors: List[List[CardSet]] = []
                            max_non_overlapping_size = 0 # maximum group size of non-overlapping tractors with size s
                            for ordering in itertools.permutations(tractor_list):
                                available_cards = CardSet(same_suit_cards)
                                tractor_combination: List[CardSet] = []
                                for tractor in ordering:
                                    if available_cards.contains(tractor.cardset):
                                        tractor_combination.append(tractor.cardset)
                                        available_cards.remove_cardset(tractor.cardset)
                                non_overlapping_tractors.append(tractor_combination)
                                max_non_overlapping_size = max(max_non_overlapping_size, len(tractor_combination))

                            # Since no combination of tractors of size s add up to the size of target, we must use all those combinations and keep searching for smaller tractors (and potentially pairs and singles).
                            if max_non_overlapping_size != target.cardset.size:
                                # How many of the tractors to use when forming the move
                                tractor_selection_count = min(max_non_overlapping_size, target.cardset.size // s)
                                reduced_cardset = target.cardset.copy()
                                reduced_cardset.draw_N_cards(tractor_selection_count * s)

                                if reduced_cardset.size > 2:
                                    reduced_move = MoveType.Tractor(reduced_cardset) # dummy CardSet
                                elif reduced_cardset.size == 2:
                                    reduced_move = MoveType.Pair(reduced_cardset.card_list()[0])
                                else:
                                    reduced_move = None
                                
                                for maximal_combo in filter(lambda combo: len(combo) == max_non_overlapping_size, non_overlapping_tractors):
                                    for selected_tractors in itertools.combinations(maximal_combo, tractor_selection_count):
                                        partial_move = CardSet() # the partially constructed move
                                        reduced_hand = hand.copy() # the remaining cards the player can choose from
                                        for tractor in selected_tractors:
                                            partial_move.add_cardset(tractor)
                                            reduced_hand.remove_cardset(tractor)
                                        
                                        if reduced_move:
                                            for match in matches_for_simple_move(reduced_move, reduced_hand):
                                                match.add_cardset(partial_move)
                                                matches.add(match)
                                        else:
                                            matches.add(partial_move)

                            else:
                                for maximal_combo in filter(lambda combo: len(combo) == max_non_overlapping_size, non_overlapping_tractors):
                                    complete_move = CardSet()
                                    for tractor in maximal_combo:
                                        complete_move.add_cardset(tractor)
                                    matches.add(complete_move)

                if not matches: # if we did not find a tractor in the same suit, the player needs to first play pairs
                    pairs = set([card for (card, count) in same_suit_cards.items() if count == 2])
                    if len(pairs) * 2 >= target.cardset.size: # If the player as at least as many pairs as the tractor size, choose any combination of the appropriate length
                        for pair_subset in itertools.combinations(pairs, target.cardset.size // 2):
                            matches.add(CardSet({p:2 for p in pair_subset}))
                
                if not matches:
                    # The player doesn't have enough pairs to add up to the length of the tractor, so they must play all those pairs, but in addition, they also need to play other cards of the same suit so that they add up to the tractor’s size.
                    must_choose_cards = {p:2 for p in pairs}
                    remaining_suit_choices = [c for c in same_suit_cards if c not in pairs]
                    if len(pairs) * 2 + len(remaining_suit_choices) >= target.cardset.size:
                        for remaining_suit_cards in itertools.combinations(remaining_suit_choices, target.cardset.size - len(pairs) * 2):
                            cardset = CardSet(must_choose_cards)
                            for card in remaining_suit_cards:
                                cardset.add_card(card)
                            matches.add(cardset)
                    else:
                        for suit_choice in remaining_suit_choices:
                            must_choose_cards[suit_choice] = 1
                
                if not matches: # The player doesn't have enough suit cards to match the tractor. So they must choose among the rest.
                    remaining_other_choices = [c for c in self.card_list() if c not in must_choose_cards]
                    records = set()
                    for remaining_other_cards in itertools.combinations(remaining_other_choices, target.cardset.size - len(pairs) * 2 - len(remaining_suit_choices)):
                        cardset = CardSet(must_choose_cards)
                        for card in remaining_other_cards:
                            cardset.add_card(card)
                        if str(cardset) in records: continue
                        records.add(str(cardset))
                        matches.add(cardset)
            
            # Sorting makes the output order deterministic
            return sorted(matches, key=lambda c: c.__str__())

        if isinstance(target, MoveType.Combo):
            largest_component = max(target.get_components(dominant_suit, dominant_rank), key=lambda c: c.cardset.size)
            remaining_target = target.cardset.copy()
            remaining_target.remove_cardset(largest_component.cardset)
            
            component_matches = matches_for_simple_move(largest_component, self)
            if remaining_target.size > 0:
                matches: Set[CardSet] = set()
                for c1_match in component_matches:
                    remaining_self = self.copy()
                    remaining_self.remove_cardset(c1_match)
                    remaining_matches = remaining_self.get_matching_moves(MoveType.Combo(remaining_target), dominant_suit, dominant_rank)
                    for remaining_match in remaining_matches:
                        combined = c1_match.copy()
                        combined.add_cardset(remaining_match)
                        matches.add(combined)
                return list(matches)
            else:
                return component_matches
        else:
            return matches_for_simple_move(target, self)
                
    def is_bigger_than(self, move: MoveType, dominant_suit: TrumpSuit, dominant_rank: int):
        "Determine if this cardset can beat a given move. If so, return the decomposition. If not, return None."
        # assert self.size == move.cardset.size, "CardSets must be the same size to compare"

        self_components = self.get_leading_moves(dominant_suit, dominant_rank)
        first_component = max(self_components, key=lambda c: c.cardset.size)
        target_suit = move.suit(dominant_suit, dominant_rank)
        self_suit = first_component.suit(dominant_suit, dominant_rank) 

        if isinstance(move, MoveType.Single) or isinstance(move, MoveType.Pair):
            if type(first_component) != type(move): return None
            self_rank = get_rank(first_component.card, dominant_suit, dominant_rank)
            target_rank = get_rank(move.card, dominant_suit, dominant_rank)
            
            if self_suit == target_suit and self_rank > target_rank or target_suit != CardSuit.TRUMP and self_suit == CardSuit.TRUMP:
                return [first_component]
        elif isinstance(move, MoveType.Tractor):
            if type(first_component) != type(move) or first_component.cardset.size < move.cardset.size: return None # You must have a tractor of equal size to beat the move (your tractor can't be longer, because your move and the target move have the same # of cards)

            self_max_rank = max([get_rank(card, dominant_suit, dominant_rank) for card in self.card_list()])
            target_max_rank = max([get_rank(card, dominant_suit, dominant_rank) for card in move.cardset.card_list()])
            if target_suit == CardSuit.TRUMP:
                # Both trump suit; compare largest rank
                if self_suit == CardSuit.TRUMP and self_max_rank > target_max_rank:
                    return [first_component]
            else:
                # If you have the same suit, you need to be bigger. Or, any trump tractor dominates any non-trump tractor of the same length.
                if self_suit == CardSuit.TRUMP or (self_suit == target_suit and self_max_rank > target_max_rank):
                    return [first_component]
        elif isinstance(move, MoveType.Combo):
            if self_suit != CardSuit.TRUMP and self_suit != target_suit: return None

            target_components = move.get_components(dominant_suit, dominant_rank)
            target_max_component = max(target_components, key=lambda c: c.cardset.size)
            possible_matches = [c for c in self_components if type(c) == type(target_max_component) and c.cardset.size == target_max_component.cardset.size]

            if not possible_matches:
                return None
            elif len(target_components) == 1:
                return [first_component] # This is the only component in the combo, and it's matched by the current player, so it's beaten
            else:
                remaining_cards = CardSet(self._cards)
                remaining_cards.remove_cardset(possible_matches[0].cardset)
                remaining_target = CardSet(move.cardset._cards)
                remaining_target.remove_cardset(target_max_component.cardset)
                rest: Union[None, List[MoveType]] = remaining_cards.is_bigger_than(MoveType.Combo(remaining_target), dominant_suit, dominant_rank)
                if rest:
                    return [first_component] + rest
                else:
                    return None
        else:
            raise AssertionError("Shouldn't get here")

    def min_rank(self, dominant_suit: TrumpSuit, dominant_rank: int):
        return min([get_rank(x, dominant_suit, dominant_rank) for x in self.card_list()])

    @classmethod
    def is_legal_combo(self, move: MoveType.Combo, other_players_cardsets: List['CardSet'], dominant_suit: TrumpSuit, dominant_rank: int):
        filtered_cardsets: List[CardSet] = [c.filter_by_suit(move.suit(dominant_suit, dominant_rank), dominant_suit, dominant_rank) for c in other_players_cardsets]
        
        def key_fn(move: MoveType):
            "Returns the maximum rank of the largest component of a move"
            return (move.cardset.size, max([get_rank(card, dominant_suit, dominant_rank) for card in move.cardset.card_list()]))
        
        decomposition = sorted(move.cardset.decompose(dominant_suit, dominant_rank), key=lambda m: m.cardset.size)
        if len(decomposition) == 1: return True, None
        for component in decomposition:
            for cardset in filtered_cardsets:
                if cardset.size < component.cardset.size: continue # If the player's total cards in the suit is less than the component, he cannot possibly suppress the combo
                moves = [c.decompose(dominant_suit, dominant_rank)[0] for c in cardset.get_matching_moves(component, dominant_suit, dominant_rank)]
                moves: List[MoveType] = [m for m in moves if type(m) == type(component) and m.cardset.size == component.cardset.size]
                
                if moves:
                    best_rank = max(map(key_fn, moves))
                    if best_rank > key_fn(component):
                        return False, min([m for m in decomposition if m.cardset.size == component.cardset.size], key=key_fn).cardset
        
        return True, None


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
            deck._cards[card] = 2
            ordering.extend([card] * 2)
        
        shuffle(ordering)
        return deck, ordering

    @classmethod
    def round_winner(self, cardsets: List['CardSet'], dominant_suit: TrumpSuit, dominant_rank: int):
        "Determine the position of the largest move."

        leading_move = MoveType.from_cardset(cardsets[0], dominant_suit, dominant_rank)
        follow_cardsets = cardsets[1:]
        if len(follow_cardsets) == 0:
            return 0

        configs = [move.is_bigger_than(leading_move, dominant_suit, dominant_rank) for move in follow_cardsets]
        
        def key_fn(move: MoveType):
            "Returns the maximum rank of the largest component of a move"
            return (move.cardset.size, max([get_rank(card, dominant_suit, dominant_rank) for card in move.cardset.card_list()]))

        ranks = [1] + [0] * len(follow_cardsets)
        for i, config in enumerate(configs, start=1):
            if config:
                best_component: MoveType = max(config, key=key_fn)
                # Add 100 to trump components so that they are definitely bigger than non-trump components.
                ranks[i] = key_fn(best_component)[1] + (100 if best_component.suit(dominant_suit, dominant_rank) == CardSuit.TRUMP else 0)
        
        return max(range(len(cardsets)), key=lambda i: ranks[i]) 

    @property
    def tensor(self):
        "Return a fixed size binary tensor of shape (108,) representing the cardset."
        rep = torch.zeros(108)
        for i, card in enumerate(ORDERING):
            if self._cards[card] >= 1:
                rep[i * 2] = 1
            if self._cards[card] == 2:
                rep[i * 2 + 1] = 1
        return rep
    
    @classmethod
    def from_tensor(self, t: torch.Tensor):
        cardset = CardSet()
        for i, card in enumerate(ORDERING):
            cardset.add_card(card, count=int(t[i * 2 : i * 2 + 2].sum()))
        return cardset

    @classmethod
    def create_deck_from_hands(self, hands: List[List[str]], kitty: List[str]):
        assert len(hands) == 4, 'There must be 4 provided hands'
        deck = []
        for card_list in hands:
            random.shuffle(card_list)
        for h1, h2, h3, h4 in zip(*hands):
            deck.extend([h1, h2, h3, h4])
        
        return deck + kitty
    
    def get_dynamic_tensor(self, trump_suit: TrumpSuit, dominant_rank: int):
        rep = torch.zeros(108)
        
        index = 0
        # First 72 or 96 cards in the order: diamonds, clubs, hearts, spades, ignoring trump cards
        for suit in [TrumpSuit.DIAMOND, TrumpSuit.CLUB, TrumpSuit.HEART, TrumpSuit.SPADE]:
            if not trump_suit.is_NT and suit == trump_suit: continue
            for numeric_rank, letter_rank in LETTER_RANK.items():
                card = letter_rank + suit
                if numeric_rank != dominant_rank:
                    if self._cards[card] >= 1:
                        rep[index] = 1
                    if self._cards[card] == 2:
                        rep[index + 1] = 1
                    index += 2
        
        # Trump suit
        if not trump_suit.is_NT:
            for numeric_rank, letter_rank in LETTER_RANK.items():
                card = letter_rank + trump_suit
                if numeric_rank != dominant_rank:
                    rep[index:index + self._cards[card]] = 1
                    index += 2
            
        # Next 8 cards are dominant rank cards
        for suit in [TrumpSuit.DIAMOND, TrumpSuit.CLUB, TrumpSuit.HEART, TrumpSuit.SPADE]:
            if trump_suit != suit:
                card = LETTER_RANK[dominant_rank] + suit
                rep[index:index + self._cards[card]] = 1
                index += 2
        
        if not trump_suit.is_NT:
            card = LETTER_RANK[dominant_rank] + trump_suit
            rep[index:index + self._cards[card]] = 1
            index += 2
            
        # Last 4 cards are the jokers
        rep[104:104 + self._cards[TrumpSuit.XJ]] = 1
        rep[106:106 + self._cards[TrumpSuit.DJ]] = 1

        return rep

    @classmethod
    def from_dynamic_tensor(self, rep: torch.Tensor, trump_suit: TrumpSuit, dominant_rank: int):
        cardset = CardSet()

        index = 0
        for suit in [TrumpSuit.DIAMOND, TrumpSuit.CLUB, TrumpSuit.HEART, TrumpSuit.SPADE]:
            if not trump_suit.is_NT and suit == trump_suit: continue
            for numeric_rank, letter_rank in LETTER_RANK.items():
                card = letter_rank + suit
                if numeric_rank != dominant_rank:
                    cardset.add_card(card, count=rep[index])
                    cardset.add_card(card, count=rep[index + 1])
                    index += 2
        
        # Trump suit
        if not trump_suit.is_NT:
            for numeric_rank, letter_rank in LETTER_RANK.items():
                card = letter_rank + trump_suit
                if numeric_rank != dominant_rank:
                    cardset.add_card(card, count=rep[index])
                    cardset.add_card(card, count=rep[index + 1])
                    index += 2
        
        # Next 8 cards are dominant rank cards
        for suit in [TrumpSuit.DIAMOND, TrumpSuit.CLUB, TrumpSuit.HEART, TrumpSuit.SPADE]:
            if suit != trump_suit:
                card = LETTER_RANK[dominant_rank] + suit
                cardset.add_card(card, count=rep[index])
                cardset.add_card(card, count=rep[index + 1])
                index += 2

        # Dominant rank and suit
        if not trump_suit.is_NT:
            card = LETTER_RANK[dominant_rank] + trump_suit
            cardset.add_card(card, count=rep[index])
            cardset.add_card(card, count=rep[index + 1])

        # Last 4 cards are the jokers
        cardset.add_card(TrumpSuit.XJ, count=rep[104] + rep[105])
        cardset.add_card(TrumpSuit.DJ, count=rep[106] + rep[107])

        return cardset

    @classmethod
    def get_tutorial_deck(self):
        """
        A tutorial deck demonstrates the most extreme cases of the game. In this example, each player owns almost an entire suit.
        """

        # Each player starts with an entire suit
        hands = [
            CardSet({
                '2♠': 2, '3♠': 2, '4♠': 2, '5♠': 2, '6♠': 2, '7♠': 2, '8♠': 2, '9♠': 2, '10♠': 2, 'J♠': 2, 'Q♠': 2, 'K♠': 2, 'A♠': 2,
                'XJ': 1,
            }),
            CardSet({
                '2♦': 2, '3♦': 2, '4♦': 2, '5♦': 2, '6♦': 2, '7♦': 2, '8♦': 2, '9♦': 2, '10♦': 2, 'J♦': 2, 'Q♦': 2, 'K♦': 2, 'A♦': 2,
                'DJ': 1,
            }),
            CardSet({
                '2♣': 2, '3♣': 2, '4♣': 2, '5♣': 2, '6♣': 2, '7♣': 2, '8♣': 2, '9♣': 2, '10♣': 2, 'J♣': 2, 'Q♣': 2, 'K♣': 2, 'A♣': 2,
                'XJ': 1,
            }),
            CardSet({
                '2♥': 2, '3♥': 2, '4♥': 2, '5♥': 2, '6♥': 2, '7♥': 2, '8♥': 2, '9♥': 2, '10♥': 2, 'J♥': 2, 'Q♥': 2, 'K♥': 2, 'A♥': 2,
                'DJ': 1,
            })
        ]
        random.shuffle(hands)

        # choose a random rank to use as kitty
        rank = random.randint(2, 14)
        kitty = CardSet({
            LETTER_RANK[rank] + CardSuit.CLUB: 2,
            LETTER_RANK[rank] + CardSuit.DIAMOND: 2,
            LETTER_RANK[rank] + CardSuit.HEART: 2,
            LETTER_RANK[rank] + CardSuit.SPADE: 2,
        })

        # Make sure no one has the kitty cards
        for hand in hands:
            hand.remove_cardset(kitty, strict=False)
        
        hands[0].add_cardset(kitty)

        # Randomly take out 8 cards from the dealer's hand to use as kitty
        kitty = hands[0].draw_N_cards(8)

        return self.create_deck_from_hands([hand.card_list() for hand in hands], kitty)
        

