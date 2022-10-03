import logging
import random
import unittest
import sys
import torch

sys.path.append('.')
from agents.RLAgents import ChaodiAgent, DeclareAgent, KittyAgent, MainAgent
from networks.Models import ChaodiModel, DeclarationModel, KittyModel, MainModel
from Simulation import Simulation
from agents.Agent import RandomAgent, SJAgent
from env.Actions import DeclareAction, DontDeclareAction, PlaceKittyAction
from env.utils import AbsolutePosition, Declaration, TrumpSuite
from env.CardSet import CardSet
from env.Game import Game

class TestGame(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(101)
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using", self.device)
        return super().setUp()

    def test_gameplay(self):
        random.seed(1235)

        game = Game()
        current_player = game.start_game()
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
        self.assertTrue(game.dealer_position == game.declarations[-1].absolute_position == AbsolutePosition.NORTH) # WEST is now the new dealer

        north_actions = game.get_observation(AbsolutePosition.NORTH).actions
        self.assertEqual(sum([a.count for a in north_actions if isinstance(a, PlaceKittyAction)]), 33) # At this point, NORTH can choose from all 33 cards

        for i in range(8):
            game.run_action(north_actions[i], AbsolutePosition.NORTH)
        
        # Skip chaodi
        north_actions = game.get_observation(AbsolutePosition.NORTH).actions
        self.assertEqual(len(north_actions), 26)

    def test_random_game_simulation(self):
        random.seed(101)
        sim = Simulation(RandomAgent('Main'), RandomAgent('Declare'), RandomAgent('Kitty'), RandomAgent('Chaodi'))
        logging.getLogger().setLevel(logging.DEBUG)
        
        while sim.step()[0]: pass
        print("Game summary:")
        print(sim.game_engine.print_status())

    def test_rl_agents(self):
        declare_model = DeclarationModel().to(self.device)
        kitty_model = KittyModel().to(self.device)
        chaodi_model = ChaodiModel().to(self.device)
        main_model = MainModel().to(self.device)
        sim = Simulation(
            main_agent=MainAgent('Main', main_model),
            declare_agent=DeclareAgent('Declare', declare_model),
            kitty_agent=KittyAgent('Kitty', kitty_model),
            chaodi_agent=ChaodiAgent('Chaodi', chaodi_model)
        )
        logging.getLogger().setLevel(logging.DEBUG)
        
        while sim.step()[0]: pass

        print(sim.game_engine.print_status())
        
        breakpoint()
        sim.declare_agent.learn_from_samples(sum(sim.declaration_history.values(), []))
        sim.kitty_agent.learn_from_samples(sum(sim.place_kitty_history.values(), []))
        sim.chaodi_agent.learn_from_samples(sum(sim.chaodi_history.values(), []))
        sim.main_agent.learn_from_samples(sum(sim.main_history.values(), []))



if __name__ == '__main__':
    unittest.main()