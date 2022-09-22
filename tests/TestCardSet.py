import unittest
import sys
import random

sys.path.append('.')
from env.utils import CardSuite, TrumpSuite
from env.CardSet import CardSet, MoveType

class TestCardSet(unittest.TestCase):

    def setUp(self) -> None:
        random.seed(123)

        self.cardset_simple = CardSet({
            '2♦': 2,
            '2♥': 2,
            '3♦': 2,
            '3♥': 2,
            '4♦': 2,
            '4♥': 2,
            'XJ': 2
        })

        self.cardset_all_suites = CardSet({
            '8♣': 1,
            '10♠': 2,
            'J♠': 2,
            'Q♠': 1,
            '2♦': 2,
            '9♦': 1,
            '10♦': 1,
            '8♥': 1,
            'DJ': 2
        })

        return super().setUp()

    def test_total_points(self):
        cardset = CardSet()
        cardset.add_card('10♣')
        cardset.add_card('7♣')
        cardset.add_card('5♠')
        cardset.add_card('5♠')
        self.assertEqual(cardset.total_points(), 20)
    
    def test_shuffle_deck(self):
        cardset, order = CardSet.new_deck()
        cardset2, order2 = cardset.new_deck()
        assert cardset.size == cardset2.size == 108
        assert len(order) == len(cardset)
        assert order != order2
    
    def test_declaration_options(self):
        fullset, _ = CardSet.new_deck()
        self.assertEqual(len(fullset.trump_declaration_options(14)), 6)

        cardset = CardSet({
            '2' + TrumpSuite.CLUB: 1,
            '2' + TrumpSuite.DIAMOND: 2,
            '2' + TrumpSuite.SPADE: 1,
            'DJ': 1,
            'XJ': 1
        })
        self.assertEqual(len(cardset.trump_declaration_options(2)), 3)
    
    def test_lead_actions_jokers(self):
        cardset = CardSet({
            'DJ': 2,
            'XJ': 2,
            '2' + TrumpSuite.DIAMOND: 2,
            '2' + TrumpSuite.CLUB: 2,
            '2' + TrumpSuite.SPADE: 2
        })
        moves1 = cardset.get_leading_moves(TrumpSuite.SPADE, 2)
        moves2 = cardset.get_leading_moves(TrumpSuite.XJ, 2)
        self.assertEqual(len(moves1), 14) # 5 singles, 5 pairs, 4 tractors
        self.assertEqual(len(moves2), 14) # 5 singles, 5 pairs, 4 tractors
    
    def test_lead_actions_skip_dominant_rank(self):
        moves = self.cardset_simple.get_leading_moves(dominant_suite=TrumpSuite.HEART, dominant_rank=3)
        self.assertEqual(len(moves), 18) # 7 singles, 7 pairs, 4 tractors
    
    def test_count_suite(self):
        # for suite in [CardSuite.CLUB, CardSuite.DIAMOND, CardSuite.HEART, CardSuite.SPADE, CardSuite.TRUMP]:
        self.assertEqual(self.cardset_simple.count_suite(CardSuite.TRUMP, TrumpSuite.HEART, 3), 10)
        self.assertEqual(self.cardset_simple.count_suite(CardSuite.HEART, TrumpSuite.HEART, 3), 0) # Since HEART is trump, there is no card in the HEART category, since they would all be trump cards
        self.assertEqual(self.cardset_simple.count_suite(CardSuite.DIAMOND, TrumpSuite.HEART, 3), 4) # 3s are trump cards, not diamond cards
        self.assertEqual(self.cardset_simple.count_suite(CardSuite.HEART, TrumpSuite.DJ, 3), 4)

    def test_matching_moves_single(self):
        # Analysis: 5♥ is a trump card in this situation, so the player's action set contains all single trump cards.
        matching_moves = self.cardset_simple.get_matching_moves(MoveType.Single('5♥'), TrumpSuite.HEART, 3)
        self.assertEqual(len(matching_moves), 5) # XJ, 3♦, 4♥, 3♥, 2♥ are the 5 valid moves
    
        # Analysis: the player has no ♣ suite cards, so they can choose any of the 7 cards to play.
        matching_moves = self.cardset_simple.get_matching_moves(MoveType.Single('5♣'), TrumpSuite.HEART, 3)
        self.assertEqual(len(matching_moves), 7)

        # Analysis: in NT mode, only the 3s and jokers are trump cards. So 3 action choices in total.
        matching_moves = self.cardset_simple.get_matching_moves(MoveType.Single('3♠'), TrumpSuite.XJ, 3)
        self.assertEqual(len(matching_moves), 3) # 3♦, 3♥, XJ

        matching_moves = self.cardset_simple.get_matching_moves(MoveType.Single('7♥'), TrumpSuite.XJ, 3)
        self.assertEqual(len(matching_moves), 2) # 4♥ and 2♥
    
    def test_matching_moves_pair(self):
        matching_moves = self.cardset_all_suites.get_matching_moves(MoveType.Pair('2♠'), TrumpSuite.HEART, 2)
        self.assertEqual(len(matching_moves), 2) # The player can play a pair of 2♦ or jokers

        matching_moves = self.cardset_all_suites.get_matching_moves(MoveType.Pair('3♠'), TrumpSuite.HEART, 2)
        self.assertEqual(len(matching_moves), 2) # Can play a pair of 10♠ or J♠

        # Analysis: the player only has 1 CLUB card. They need play 1 additional card. There are 8 to choose from.
        matching_moves = self.cardset_all_suites.get_matching_moves(MoveType.Pair('3♣'), TrumpSuite.HEART, 2)
        self.assertEqual(len(matching_moves), 8)

        # Analysis: the player has no CLUB cards. So they can pick any two cards to play. If they play a trump pair, it's a RUFF so it's counted as a pair. If they choose any two other cards, it's counted as a passive combo.
        matching_moves = self.cardset_simple.get_matching_moves(MoveType.Pair('3♣'), TrumpSuite.HEART, 2)
        self.assertEqual(len(matching_moves), 28) # 7 pairs + 21 two card combos
    
    def test_matching_moves_tractor(self):
        matching_moves = self.cardset_simple.get_matching_moves(MoveType.Tractor('5♥', '6♥', CardSet({'5♥': 2, '6♥': 2})), TrumpSuite.HEART, 3)
        self.assertEqual(len(matching_moves), 3) # Either [2♥ 2♥ 4♥ 4♥], [3♦ 3♦ 3♥ 3♥], or [3♥ 3♥ XJ XJ]

        matching_moves = self.cardset_simple.get_matching_moves(MoveType.Tractor('5♠', '6♠', CardSet({'5♠': 2, '6♠': 2})), TrumpSuite.HEART, 3)
        print(len(matching_moves), matching_moves) # 161

if __name__ == '__main__':
    unittest.main()