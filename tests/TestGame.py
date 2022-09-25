import logging
import random
from time import sleep
import unittest
import sys

sys.path.append('.')
from Simulation import Simulation
from agents.Agent import RandomAgent, SJAgent
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
        # print("Start player:", current_player)
        for _ in range(99):
            game.run_action(DontDeclareAction(), current_player)
            current_player = current_player.next_position

        # Test action list
        north_actions = game.get_observation(AbsolutePosition.NORTH).actions
        west_actions = game.get_observation(AbsolutePosition.WEST).actions
        # print('West actions:', west_actions) 
        # print('North actions:', north_actions) # NORTH has double red jokers in this game at action 1.
        self.assertEqual(len(west_actions), 2)
        self.assertEqual(len(north_actions), 3)
        game.run_action(west_actions[0], AbsolutePosition.WEST) # West declares something             
        game.run_action(north_actions[0], AbsolutePosition.NORTH) # North overrides the declaration with double red joker
        
        # Last player draws a card
        game.run_action(DontDeclareAction(), current_player)
        # print('Current declaration:', game.declaration)

        self.assertEqual(sum([hand.size for hand in game.hands.values()]), 108) # all cards should be distributed
        self.assertEqual(game.hands[game.dealer_position].size, 33) # Dealer has 25 + 8 = 33 cards
        self.assertTrue(all([hand.size == 25 for player, hand in game.hands.items() if player != game.dealer_position]))
        self.assertTrue(game.dealer_position == game.declaration.absolute_position == AbsolutePosition.NORTH) # WEST is now the new dealer

        north_actions = game.get_observation(AbsolutePosition.NORTH).actions
        self.assertEqual(sum([a.count for a in north_actions if isinstance(a, PlaceKittyAction)]), 33) # At this point, NORTH can choose from all 33 cards

        for i in range(8):
            game.run_action(north_actions[i], AbsolutePosition.NORTH)
        
        # Skip chaodi
        north_actions = game.get_observation(AbsolutePosition.NORTH).actions
        self.assertEqual(len(north_actions), 26)

    def test_random_game_simulation(self):
        random.seed(101)
        sim = Simulation(RandomAgent('NS'), RandomAgent('WE'))
        logging.getLogger().setLevel(logging.DEBUG)
        
        for _ in range(100):
            sim.step()

        while sim.step()[0]: pass
        print("Game summary:")
        print(sim.game_engine.print_status())



if __name__ == '__main__':
    unittest.main()