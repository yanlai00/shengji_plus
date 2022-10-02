# Represent a set of cards in a Shengji game.
import json
from random import shuffle
import random
from typing import Dict, List, Set, Tuple, Union
import torch
import itertools

from env.utils import LETTER_RANK, ORDERING, CardSuite, TrumpSuite, get_rank, get_suite

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
    def suite(self, dominant_suite: TrumpSuite, dominant_rank: int) -> CardSuite:
        "Returns the suite of the move"
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
        @property
        def cardset(self):
            return CardSet({self.card: 2})
        def suite(self, dominant_suite: TrumpSuite, dominant_rank: int) -> CardSuite:
            return get_suite(self.card, dominant_suite, dominant_rank)
    
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
        def suite(self, dominant_suite: TrumpSuite, dominant_rank: int) -> CardSuite:
            return get_suite(self._cardset.card_list()[0], dominant_suite, dominant_rank)
    
    class Combo(MoveType):
        "An active combo move. All elements in a combo are required to have the same CardSuite."
        def __init__(self, cardset: 'CardSet') -> None:
            self._cardset = cardset
        def __repr__(self) -> str:
            return f"Combo({self._cardset})"
        @property
        def multiplier(self):
            print("Do not call .multiplier of combo directly! Use .get_multiplier(...) instead.")
            return 2
        def get_components(self, dominant_suite: TrumpSuite, dominant_rank: int):
            return self._cardset.decompose(dominant_suite, dominant_rank)
        def get_multiplier(self, dominant_suite: TrumpSuite, dominant_rank: int):
            return max([c.multiplier for c in self.get_components(dominant_suite, dominant_rank)])
        @property
        def cardset(self):
            return self._cardset
        def suite(self, dominant_suite: TrumpSuite, dominant_rank: int) -> CardSuite:
            return self.get_components(dominant_suite, dominant_rank)[0].suite(dominant_suite, dominant_rank)

    @classmethod
    def from_cardset(self, cardset: 'CardSet', dominant_suite: TrumpSuite, dominant_rank: int):
        components = cardset.decompose(dominant_suite, dominant_rank)
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
    
    def filter_by_suite(self, suite: CardSuite, dominant_suite: TrumpSuite, dominant_rank: int):
        subset = CardSet()
        for card, count in self._cards.items():
            if count > 0 and get_suite(card, dominant_suite, dominant_rank) == suite:
                subset.add_card(card, count)
        return subset

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
    
    def add_cardset(self, cardset: 'CardSet'):
        for card in cardset.card_list():
            self.add_card(card) 
    
    def draw_N_cards(self, N: int):
        assert N <= self.size, "Cannot draw more cards than there are available"
        ordering = []
        for card, count in self._cards.items():
            ordering.extend([card] * count)
        
        random.shuffle(ordering)
        selected_cards = ordering[:N]
        for card in selected_cards:
            self.remove_card(card)
        
        return selected_cards
        
    def remove_card(self, card: str, count=1):
        assert self._cards[card] >= count, "Cannot play the given card"
        if self._cards[card] >= count:
            self._cards[card] -= count
        else:
            print(f"WARNING: attempting to remove a non-existent card ({card}) from cardset")
    
    def remove_cardset(self, cardset: 'CardSet'):
        for card in cardset.card_list():
            self.remove_card(card)
    
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
        """Find all the possible trump suites that the user can declare, as well as their level. Higher level declarations override lower ones.
        
        Pair of same suite dominant rank cards: level 1
        Pair of b&w jokers: level 2
        Pair of color jokers: level 3
        """
        options: Dict[TrumpSuite, int] = {}
        if self._cards["DJ"] == 2:
            options[TrumpSuite.DJ] = 3
        if self._cards["XJ"] == 2:
            options[TrumpSuite.XJ] = 2
        
        letter_rank = LETTER_RANK[rank]
        for suite in [TrumpSuite.CLUB, TrumpSuite.DIAMOND, TrumpSuite.HEART, TrumpSuite.SPADE]:
            if self._cards[letter_rank + suite] >= 1:
                options[suite] = self._cards[letter_rank + suite] - 1
        
        return options
    
    def get_count(self, suite: TrumpSuite, rank: int):
        if suite == TrumpSuite.DJ or suite == TrumpSuite.XJ:
            return self._cards['DJ'] + self._cards['XJ']
        return self._cards[LETTER_RANK[rank] + suite]

    def get_leading_moves(self, dominant_suite: TrumpSuite, dominant_rank: int, include_combos = False):
        "Return all possible move actions the player can make if they lead the trick."
        moves: List[MoveType] = []
        suite_list = [CardSuite.CLUB, CardSuite.DIAMOND, CardSuite.HEART, CardSuite.SPADE, CardSuite.TRUMP]
        
        if include_combos:
            cards_by_suite = {suite: CardSet() for suite in suite_list}
            for card in self.card_list():
                cards_by_suite[get_suite(card, dominant_suite, dominant_rank)].add_card(card)
            for suite, suite_cardset in cards_by_suite.items():
                records = set()
                for size in range(1, suite_cardset.size + 1):
                    for combo in itertools.combinations(suite_cardset.card_list(), size):
                        combo_cardset = CardSet()
                        for move in combo:
                            combo_cardset.add_card(move)
                        components = combo_cardset.decompose(dominant_suite, dominant_rank)
                        if combo_cardset not in records:
                            records.add(combo_cardset)
                            moves.append(MoveType.Combo(combo_cardset) if len(components) > 1 else components[0])
            
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
        if not dominant_suite.is_NT:
            suite_list.remove(dominant_suite)
        
        pairs_by_suite = {suite: [] for suite in suite_list}
        for card in pair_cards:
            pairs_by_suite[get_suite(card, dominant_suite, dominant_rank)].append(card)

        for suite, cards in pairs_by_suite.items():
            card_ranks: List[Tuple[int, str]] = []
            for card in cards:
                rank = get_rank(card, dominant_suite, dominant_rank)
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
    
    def decompose(self, dominant_suite: TrumpSuite, dominant_rank: int) -> List[Union[MoveType.Single, MoveType.Pair, MoveType.Tractor]]:
        "Decompose a leading move into components using a greedy approach."
        remaining_cards = CardSet(self._cards)
        components = []
        while remaining_cards.size > 0:
            largest = max(remaining_cards.get_leading_moves(dominant_suite, dominant_rank), key=lambda move: move.cardset.size)
            remaining_cards.remove_cardset(largest.cardset)
            components.append(largest)
        
        return components

    def get_matching_moves(self, target: MoveType, dominant_suite: TrumpSuite, dominant_rank: int):
        target_suite = target.suite(dominant_suite, dominant_rank) # We need to first figure out the suite of the move the player is following.
        
        # Convenient helper function that returns a list of longest tractors in a CardSet
        def get_longest_tractors(cardset: CardSet) -> Tuple[int, List[MoveType.Tractor]]:
            tractors = sorted([t for t in cardset.get_leading_moves(dominant_suite, dominant_rank) if isinstance(t, MoveType.Tractor) and t.suite(dominant_suite, dominant_rank) == target_suite], key=lambda t: t.cardset.size, reverse=True)
            
            if not tractors: return 0, []

            longest_tractors = []
            for t in tractors:
                if t.cardset.size == tractors[0].cardset.size:
                    longest_tractors.append(t)
            return tractors[0].cardset.size, longest_tractors

        def matches_for_simple_move(target: Union[MoveType.Single, MoveType.Pair, MoveType.Tractor], hand: CardSet):
            matches: Set[CardSet] = set()
            same_suite_cards = {card:v for card,v in hand._cards.items() if v > 0 and get_suite(card, dominant_suite, dominant_rank) == target_suite}
            suite_count = hand.count_suite(target_suite, dominant_suite, dominant_rank)

            if isinstance(target, MoveType.Single):
                if len(same_suite_cards) > 0:
                    for card in same_suite_cards:
                        matches.add(CardSet({card: 1}))
                else:
                    for card, count in hand._cards.items():
                        if count > 0:
                            matches.add(CardSet({card: 1}))
            elif isinstance(target, MoveType.Pair):
                if suite_count > len(same_suite_cards): # If total count is greater than variety count, then there must be a pair
                    for card, count in same_suite_cards.items():
                        if count == 2:
                            matches.add(CardSet({card: 2}))
                elif suite_count >= 2:
                    for c1, c2 in itertools.combinations(same_suite_cards, 2):
                        matches.add(CardSet({c1: 1, c2: 1}))
                elif suite_count == 1:
                    fixed_card, _ = same_suite_cards.popitem()
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
                if suite_count >= target.cardset.size:
                    # If there are enough cards in the suite, enumerate all possible tractors and check if the player has them
                    
                    exact_tractors = [t.cardset for t in hand.get_leading_moves(dominant_suite, dominant_rank) if isinstance(t, MoveType.Tractor) and t.suite(dominant_suite, dominant_rank) == target_suite and t.cardset.size == target.cardset.size]
                    matches.update(exact_tractors)
                    
                    if not exact_tractors:                        
                        s, tractor_list = get_longest_tractors(CardSet(same_suite_cards))
                        if s > 0:
                            # Find all non-overlapping combinations of tractors
                            non_overlapping_tractors: List[List[CardSet]] = []
                            max_non_overlapping_size = 0 # maximum group size of non-overlapping tractors with size s
                            for ordering in itertools.permutations(tractor_list):
                                available_cards = CardSet(same_suite_cards)
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
                                reduced_cardset = CardSet(target.cardset._cards)
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
                                        reduced_hand = CardSet(hand._cards) # the remaining cards the player can choose from
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

                if not matches: # if we did not find a tractor in the same suite, the player needs to first play pairs
                    pairs = set([card for (card, count) in same_suite_cards.items() if count == 2])
                    if len(pairs) * 2 >= target.cardset.size: # If the player as at least as many pairs as the tractor size, choose any combination of the appropriate length
                        for pair_subset in itertools.combinations(pairs, target.cardset.size // 2):
                            matches.add(CardSet({p:2 for p in pair_subset}))
                
                if not matches:
                    # The player doesn't have enough pairs to add up to the length of the tractor, so they must play all those pairs, but in addition, they also need to play other cards of the same suite so that they add up to the tractor’s size.
                    must_choose_cards = {p:2 for p in pairs}
                    remaining_suite_choices = [c for c in same_suite_cards if c not in pairs]
                    if len(pairs) * 2 + len(remaining_suite_choices) >= target.cardset.size:
                        for remaining_suite_cards in itertools.combinations(remaining_suite_choices, target.cardset.size - len(pairs) * 2):
                            cardset = CardSet(must_choose_cards)
                            for card in remaining_suite_cards:
                                cardset.add_card(card)
                            matches.add(cardset)
                    else:
                        for suite_choice in remaining_suite_choices:
                            must_choose_cards[suite_choice] = 1
                
                if not matches: # The player doesn't have enough suite cards to match the tractor. So they must choose among the rest.
                    remaining_other_choices = [c for c in self.card_list() if c not in must_choose_cards]
                    records = set()
                    for remaining_other_cards in itertools.combinations(remaining_other_choices, target.cardset.size - len(pairs) * 2 - len(remaining_suite_choices)):
                        cardset = CardSet(must_choose_cards)
                        for card in remaining_other_cards:
                            cardset.add_card(card)
                        if str(cardset) in records: continue
                        records.add(str(cardset))
                        matches.add(cardset)
            
            # Sorting makes the output order deterministic
            return sorted(matches, key=lambda c: c.__str__())

        if isinstance(target, MoveType.Combo):
            largest_component = max(target.get_components(dominant_suite, dominant_rank), key=lambda c: c.cardset.size)
            remaining_target = CardSet(target.cardset._cards)
            remaining_target.remove_cardset(largest_component.cardset)
            
            component_matches = matches_for_simple_move(largest_component, self)
            if remaining_target.size > 0:
                matches: List[CardSet] = []
                for c1_match in component_matches:
                    remaining_self = CardSet(self._cards)
                    remaining_self.remove_cardset(c1_match)
                    remaining_matches = remaining_self.get_matching_moves(MoveType.Combo(remaining_target), dominant_suite, dominant_rank)
                    for remaining_match in remaining_matches:
                        combined = CardSet(c1_match._cards)
                        combined.add_cardset(remaining_match)
                        matches.append(combined)
                return matches
            else:
                return component_matches
        else:
            return matches_for_simple_move(target, self)
                
    def is_bigger_than(self, move: MoveType, dominant_suite: TrumpSuite, dominant_rank: int):
        "Determine if this cardset can beat a given move. If so, return the decomposition. If not, return None."
        assert self.size == move.cardset.size, "CardSets must be the same size to compare"

        self_components = self.get_leading_moves(dominant_suite, dominant_rank)
        first_component = max(self_components, key=lambda c: c.cardset.size)
        target_suite = move.suite(dominant_suite, dominant_rank)
        self_suite = first_component.suite(dominant_suite, dominant_rank) 

        if isinstance(move, MoveType.Single) or isinstance(move, MoveType.Pair):
            if type(first_component) != type(move): return None
            self_rank = get_rank(first_component.card, dominant_suite, dominant_rank)
            target_rank = get_rank(move.card, dominant_suite, dominant_rank)
            
            if self_suite == target_suite and self_rank > target_rank or target_suite != CardSuite.TRUMP and self_suite == CardSuite.TRUMP:
                return [first_component]
        elif isinstance(move, MoveType.Tractor):
            if type(first_component) != type(move) or first_component.cardset.size < move.cardset.size: return None # You must have a tractor of equal size to beat the move (your tractor can't be longer, because your move and the target move have the same # of cards)

            self_max_rank = max([get_rank(card, dominant_suite, dominant_rank) for card in self.card_list()])
            target_max_rank = max([get_rank(card, dominant_suite, dominant_rank) for card in move.cardset.card_list()])
            if target_suite == CardSuite.TRUMP:
                # Both trump suite; compare largest rank
                if self_suite == CardSuite.TRUMP and self_max_rank > target_max_rank:
                    return [first_component]
            else:
                # If you have the same suite, you need to be bigger. Or, any trump tractor dominates any non-trump tractor of the same length.
                if self_suite == CardSuite.TRUMP or (self_suite == target_suite and self_max_rank > target_max_rank):
                    return [first_component]
        elif isinstance(move, MoveType.Combo):
            if self_suite != CardSuite.TRUMP and self_suite != target_suite: return None

            target_components = move.get_components(dominant_suite, dominant_rank)
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
                rest: Union[None, List[MoveType]] = remaining_cards.is_bigger_than(MoveType.Combo(remaining_target), dominant_suite, dominant_rank)
                if rest:
                    return [first_component] + rest
                else:
                    return None
        else:
            raise AssertionError("Shouldn't get here")

    @classmethod
    def is_legal_combo(self, move: MoveType.Combo, other_players_cardsets: List['CardSet'], dominant_suite: TrumpSuite, dominant_rank: int):
        filtered_cardsets: List[CardSet] = [c.filter_by_suite(move.suite(dominant_suite, dominant_rank), dominant_suite, dominant_rank) for c in other_players_cardsets]
        
        def key_fn(move: MoveType):
            "Returns the maximum rank of the largest component of a move"
            return (move.cardset.size, max([get_rank(card, dominant_suite, dominant_rank) for card in move.cardset.card_list()]))
        
        decomposition = sorted(move.cardset.decompose(dominant_suite, dominant_rank), key=lambda m: m.cardset.size)
        if len(decomposition) == 1: return True, None
        for component in decomposition:
            for cardset in filtered_cardsets:
                if cardset.size < component.cardset.size: continue # If the player's total cards in the suite is less than the component, he cannot possibly suppress the combo
                moves = [c.decompose(dominant_suite, dominant_rank)[0] for c in cardset.get_matching_moves(component, dominant_suite, dominant_rank)]
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
    def round_winner(self, cardsets: List['CardSet'], dominant_suite: TrumpSuite, dominant_rank: int):
        "Determine the position of the largest move."

        leading_move = MoveType.from_cardset(cardsets[0], dominant_suite, dominant_rank)
        follow_cardsets = cardsets[1:]
        if len(follow_cardsets) == 0:
            return 0

        configs = [move.is_bigger_than(leading_move, dominant_suite, dominant_rank) for move in follow_cardsets]
        
        def key_fn(move: MoveType):
            "Returns the maximum rank of the largest component of a move"
            return (move.cardset.size, max([get_rank(card, dominant_suite, dominant_rank) for card in move.cardset.card_list()]))

        ranks = [1] + [0] * len(follow_cardsets)
        for i, config in enumerate(configs, start=1):
            if config:
                best_component: MoveType = max(config, key=key_fn)
                # Add 100 to trump components so that they are definitely bigger than non-trump components.
                ranks[i] = key_fn(best_component)[1] + (100 if best_component.suite(dominant_suite, dominant_rank) == CardSuite.TRUMP else 0)
        
        return max(range(len(cardsets)), key=lambda i: ranks[i]) 

    @property
    def tensor(self):
        "Return a fixed size binary tensor representing the cardset."
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

