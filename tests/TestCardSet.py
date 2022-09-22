import unittest
import sys
import random

sys.path.append('.')
from env.utils import TrumpSuite
from env.CardSet import CardSet

class TestCardSet(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(123)
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

        cardset = CardSet()
        cardset.add_card('2' + TrumpSuite.CLUB)
        cardset.add_card('2' + TrumpSuite.DIAMOND)
        cardset.add_card('2' + TrumpSuite.DIAMOND)
        cardset.add_card('2' + TrumpSuite.SPADE)
        cardset.add_card('DJ')
        cardset.add_card('XJ')
        self.assertEqual(len(cardset.trump_declaration_options(2)), 3)
    
    def test_lead_actions(self):
        cardset = CardSet({
            'DJ': 2,
            'XJ': 2,
            '2' + TrumpSuite.DIAMOND: 2,
            '2' + TrumpSuite.CLUB: 2,
            '2' + TrumpSuite.SPADE: 2
        })
        moves1 = cardset.get_leading_moves(2, TrumpSuite.SPADE)
        moves2 = cardset.get_leading_moves(2, TrumpSuite.XJ)
        print(f"2{TrumpSuite.SPADE}:", moves1)
        print('NT:', moves2)
    
    def test_lead_actions_skip_dominant_rank(self):
        cardset = CardSet({
            '2' + TrumpSuite.DIAMOND: 2,
            '2' + TrumpSuite.HEART: 2,
            '3' + TrumpSuite.DIAMOND: 2,
            '3' + TrumpSuite.HEART: 2,
            '4' + TrumpSuite.DIAMOND: 2,
            '4' + TrumpSuite.HEART: 2
        })

        moves = cardset.get_leading_moves(dominant_rank=3, dominant_suite=TrumpSuite.HEART)
        print(moves)



if __name__ == '__main__':
    unittest.main()