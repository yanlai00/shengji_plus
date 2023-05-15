# Need to rewrite this file

import unittest
import sys
import random

sys.path.append('.')
from env.utils import CardSuit, TrumpSuit
from env.CardSet import CardSet, Move
from game_consts import TOTAL_CARDS

class TestCardSet(unittest.TestCase):

    def setUp(self) -> None:
        random.seed(123)

        self.cardset_simple = CardSet({
            '2♦': 1,
            '2♥': 1,
            '3♦': 1,
            '3♥': 1,
            '4♦': 1,
            '4♥': 1,
        })

        self.cardset_all_suits = CardSet({
            '8♣': 1,
            '10♠': 1,
            'J♠': 1,
            'Q♠': 1,
            '2♦': 1,
            '9♦': 1,
            '10♦': 1,
            '8♥': 1,
        })

        self.cardset_big_suit = CardSet({
            'A♥': 1,
            'K♥': 1,
            'Q♥': 1,
            'J♥': 1,
            '10♥': 1,
            '4♥': 1,
            '2♥': 1,
            '3♣': 1,
        })

        self.cardset_multiple_small_tractors = CardSet({
            'K♦': 1,
            'Q♦': 1,
            '10♦': 1,
            '9♦': 1,
            '7♦': 1,
            '6♦': 1
        })

        return super().setUp()
    
    def test_shuffle_deck(self):
        cardset, order = CardSet.new_deck()
        cardset2, order2 = cardset.new_deck()
        assert cardset.size == cardset2.size == TOTAL_CARDS
        assert len(order) == len(cardset)
        assert order != order2
    
    def test_lead_actions_jokers(self):
        cardset = CardSet({
            '2' + TrumpSuit.DIAMOND: 2,
            '2' + TrumpSuit.CLUB: 2,
            '2' + TrumpSuit.SPADE: 2
        })
        moves1 = cardset.get_leading_moves()
        self.assertEqual(len(moves1), 19) # 5 singles, 5 pairs, 4 length-2 tractors, 3 length-3 tractors, 2 length-4 tractors
    
    def test_count_suit(self):
        # for suit in [CardSuit.CLUB, CardSuit.DIAMOND, CardSuit.HEART, CardSuit.SPADE, CardSuit.TRUMP]:
        self.assertEqual(self.cardset_simple.count_suit(CardSuit.TRUMP, TrumpSuit.HEART, 3), 10)
        self.assertEqual(self.cardset_simple.count_suit(CardSuit.HEART, TrumpSuit.HEART, 3), 0) # Since HEART is trump, there is no card in the HEART category, since they would all be trump cards
        self.assertEqual(self.cardset_simple.count_suit(CardSuit.DIAMOND, TrumpSuit.HEART, 3), 4) # 3s are trump cards, not diamond cards
        self.assertEqual(self.cardset_simple.count_suit(CardSuit.HEART, TrumpSuit.NT, 3), 4)

    def test_matching_moves_single(self):
        # Analysis: 5♥ is a trump card in this situation, so the player's action set contains all single trump cards.
        matching_moves = self.cardset_simple.get_matching_moves(Move('5♥'), TrumpSuit.HEART, 3)
        self.assertEqual(len(matching_moves), 4) # 3♦, 4♥, 3♥, 2♥ are the 5 valid moves
    
        # Analysis: the player has no ♣ suit cards, so they can choose any of the 7 cards to play.
        matching_moves = self.cardset_simple.get_matching_moves(Move('5♣'), TrumpSuit.HEART, 3)
        self.assertEqual(len(matching_moves), 7)

        # Analysis: in NT mode, only the 3s and jokers are trump cards. So 3 action choices in total.
        matching_moves = self.cardset_simple.get_matching_moves(Move('3♠'), TrumpSuit.NT, 3)
        self.assertEqual(len(matching_moves), 2) # 3♦, 3♥

        matching_moves = self.cardset_simple.get_matching_moves(Move('7♥'), TrumpSuit.NT, 3)
        self.assertEqual(len(matching_moves), 2) # 4♥ and 2♥
    
    def test_cardset_compare_single(self):
        # Move card can beat another single card if the rank is higher, or if it's a trump card but the other one is not
        single_k_move = Move('K♦')
        single_ace = CardSet({'A♦': 1})
        single_queen = CardSet({'Q♦': 1})
        single_jack_heart = CardSet({'J♥': 1})
        single_k_heart = CardSet({'K♥': 1})
        
        # Same suit cards. Compare rank only.
        self.assertTrue(all([single_ace.is_bigger_than(single_k_move, suit, 2) for suit in (TrumpSuit.CLUB, TrumpSuit.DIAMOND, TrumpSuit.NT, TrumpSuit.HEART, TrumpSuit.SPADE)]))
        self.assertFalse(all([single_queen.is_bigger_than(single_k_move, suit, 2) for suit in (TrumpSuit.CLUB, TrumpSuit.DIAMOND, TrumpSuit.NT, TrumpSuit.HEART, TrumpSuit.SPADE)]))

        # Trump against non-trump (ruff)
        self.assertTrue(single_jack_heart.is_bigger_than(single_k_move, TrumpSuit.HEART, 2))

        # Non-trump against non-trump of different suit
        self.assertFalse(single_jack_heart.is_bigger_than(single_k_move, TrumpSuit.CLUB, 2))

        # Dominant rank is larger than other non-joker trump cards
        self.assertTrue(single_queen.is_bigger_than(single_k_move, TrumpSuit.DIAMOND, 12))

        # Dominant card should be larger than other dominant-rank cards
        self.assertTrue(single_k_heart.is_bigger_than(single_k_move, TrumpSuit.HEART, 13))
        self.assertFalse(single_k_heart.is_bigger_than(single_k_move, TrumpSuit.NT, 13))
        self.assertFalse(single_k_heart.is_bigger_than(single_k_move, TrumpSuit.NT, 12))

    def test_multi_compare_combo(self):
        # Player0 hands leading position to player2 in an NT game
        player0_cards = CardSet({'10♥': 1})
        player1_cards = CardSet({'A♥': 1}) # beats player0
        player2_cards = CardSet({'2♦': 1}) # Ruff
        player3_cards = CardSet({'4♥': 1}) # follows
        self.assertEqual(CardSet.round_winner([player0_cards, player1_cards, player2_cards, player3_cards], TrumpSuit.NT, 2), 2)

        # Player with the largest trump pair wins (given that they play 5 trump cards containing 2 pairs)
        player0_cards = CardSet({ 'A♥': 2, 'K♥': 1, 'Q♥': 2 }) # Combo with 2 pairs and 1 single
        player1_cards = CardSet({ '5♥': 2, '3♥': 1, '7♥': 1, 'J♥': 1 }) # Match suit
        player2_cards = CardSet({ '6♠': 2, '7♠': 2, '2♣': 1 }) # Uses a tractor to ruff the combo
        player3_cards = CardSet({ 'A♠': 2, '5♠': 2, '10♠': 1 }) # Beats player2 because only the largest pair matters here.
        self.assertEqual(CardSet.round_winner([player0_cards, player1_cards, player2_cards, player3_cards], TrumpSuit.SPADE, 2), 3)

        # Player with the largest tractor wins
        player0_cards = CardSet({ '9♥': 2, '10♥': 2 }) # 1 tractor
        player1_cards = CardSet({ 'A♥': 2, 'K♥': 2 }) # Beats player0 in same suit
        player2_cards = CardSet({ '6♠': 2, '7♠': 2 }) # Ruffs using a tractor
        player3_cards = CardSet({ '2♠': 2, '2♣': 2 }) # Ruffs using bigger tractor
        self.assertEqual(CardSet.round_winner([player0_cards, player1_cards, player2_cards, player3_cards], TrumpSuit.SPADE, 2), 3)

        # Example where player0 is the biggest
        player0_cards = CardSet({'Q♥': 2})
        player1_cards = CardSet({'3♥': 2})
        player2_cards = CardSet({ '10♠': 1, 'J♥': 1 }) # player2 escapes 10 points in trump suit
        player3_cards = CardSet({ '5♥': 1, '9♥': 1 })
        self.assertEqual(CardSet.round_winner([player0_cards, player1_cards, player2_cards, player3_cards], TrumpSuit.SPADE, 2), 0)

    def test_tensor_conversion(self):
        t1 = self.cardset_simple.tensor
        self.assertEqual(self.cardset_simple, CardSet.from_tensor(t1))

        t2 = self.cardset_all_suits.tensor
        self.assertEqual(self.cardset_all_suits, CardSet.from_tensor(t2))

    def test_trump_suit_tensor(self):
        self.assertEqual(TrumpSuit.from_tensor(TrumpSuit.DIAMOND.tensor), TrumpSuit.DIAMOND)
        self.assertEqual(TrumpSuit.from_tensor(TrumpSuit.CLUB.tensor), TrumpSuit.CLUB)
        self.assertEqual(TrumpSuit.from_tensor(TrumpSuit.HEART.tensor), TrumpSuit.HEART)
        self.assertEqual(TrumpSuit.from_tensor(TrumpSuit.SPADE.tensor), TrumpSuit.SPADE)
        self.assertEqual(TrumpSuit.from_tensor(TrumpSuit.NT.tensor), TrumpSuit.NT)
    
    def test_dynamic_tensor(self):
        full_deck, order = CardSet.new_deck()

        for suit in [TrumpSuit.DIAMOND, TrumpSuit.CLUB, TrumpSuit.NT]:
            for rank in range(2, 15):
                self.assertEqual(CardSet.from_dynamic_tensor(full_deck.get_dynamic_tensor(suit, rank), suit, rank), full_deck)
        
        half_deck = CardSet()
        for card in order[:28]:
            half_deck.add_card(card)
        
        for suit in [TrumpSuit.HEART, TrumpSuit.SPADE, TrumpSuit.NT]:
            for rank in range(2, 15):
                self.assertEqual(CardSet.from_dynamic_tensor(half_deck.get_dynamic_tensor(suit, rank), suit, rank), half_deck)
    
    def test_random_hand(self):
        random.seed(10)
        full_deck, order = CardSet.new_deck()
        cardset = CardSet()
        for card in order[:25]:
            cardset.add_card(card)
        print(cardset)

if __name__ == '__main__':
    unittest.main()