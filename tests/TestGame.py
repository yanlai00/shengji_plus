import random
import unittest
import sys

sys.path.append('.')
from agents.Agent import SJAgent
from env.Actions import DeclareAction, DontDeclareAction, PlaceKittyAction
from env.utils import AbsolutePosition, Declaration, TrumpSuite
from env.CardSet import CardSet
from env.Game import Game

class TestGame(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(123)
        return super().setUp()

    def test_gameplay(self):
        random.seed(1235)

        game = Game()
        current_player = AbsolutePosition(max(game.hands.keys(), key=lambda x: game.hands[x].size))
        print("Start player:", current_player)
        for _ in range(99):
            game.run_action(DontDeclareAction(), current_player)
            current_player = current_player.next_position

        # Test action list
        west_player = SJAgent(AbsolutePosition.WEST)
        north_player = SJAgent(AbsolutePosition.NORTH)
        north_actions = north_player.get_actions(game.get_observation(AbsolutePosition.NORTH))
        west_actions = west_player.get_actions(game.get_observation(AbsolutePosition.WEST))
        print('West actions:', west_actions) 
        print('North actions:', north_actions) # NORTH has double red jokers in this game at action 1.
        self.assertEqual(len(west_actions), 2)
        self.assertEqual(len(north_actions), 3)
        game.run_action(west_actions[0], west_player.position) # West declares something             
        game.run_action(north_actions[0], north_player.position) # North overrides the declaration with double red joker
        
        # Last player draws a card
        game.run_action(DontDeclareAction(), current_player)
        print('Current declaration:', game.declaration)

        self.assertEqual(sum([hand.size for hand in game.hands.values()]), 108) # all cards should be distributed
        self.assertEqual(game.hands[game.dealer_position].size, 33) # Dealer has 25 + 8 = 33 cards
        self.assertTrue(all([hand.size == 25 for player, hand in game.hands.items() if player != game.dealer_position]))
        self.assertTrue(game.dealer_position == game.declaration.absolute_position == AbsolutePosition.NORTH) # WEST is now the new dealer

        north_actions = north_player.get_actions(game.get_observation(AbsolutePosition.NORTH))
        self.assertEqual(sum([a.count for a in north_actions if isinstance(a, PlaceKittyAction)]), 33) # At this point, NORTH can choose from all 33 cards
        print("Initial actions", north_actions)

        for i in range(8):
            game.run_action(north_actions[i], north_player.position)
        
        north_actions = north_player.get_actions(game.get_observation(AbsolutePosition.NORTH))
        print("New actions", north_actions)






if __name__ == '__main__':
    unittest.main()