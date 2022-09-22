# Describes the current state of a Shengji game. Obeys Markov's property.

import random
from typing import List, Tuple, Union
from env.Actions import Action, DeclareAction, DontChaodiAction, DontDeclareAction, PlaceKittyAction

from env.Observation import Observation
from env.utils import *
from .CardSet import CardSet

class Game:
    def __init__(self, dominant_rank=2, dealer_position: AbsolutePosition = None, enable_chaodi = False) -> None:
        # Player information
        self.hands = {
            AbsolutePosition.NORTH: CardSet(),
            AbsolutePosition.SOUTH: CardSet(),
            AbsolutePosition.WEST: CardSet(),
            AbsolutePosition.EAST: CardSet()
        }
        self.deck = iter(CardSet.new_deck()[1]) # First 100 are drawn, last 8 given to dealer.
        self.kitty = CardSet() # dealer puts the kitty in 8 rounds, one card per round

        # Public game information
        self.round_history: List[Tuple[AbsolutePosition, Tuple[CardSet]]] = []
        "A list of past tricks, each having the structure (P, (CardSet, CardSet, CardSet, CardSet)), where P is one of 'N', 'W', 'S', or 'E', and CardSets are in the order they were played (so for instance, element 0 is played by P)."
        
        self.draw_stage_completed = False # True if all 100 cards are drawn and everyone had their chance to declare/redeclare.
        self.kitty_stage_completed = False # True if the kitty is fixed.
        self.dominant_rank = dominant_rank
        self.declaration: Declaration = None # Information about who declared the trump suite, what it is, and its level
        self.current_declaration_turn: AbsolutePosition = None # Once a player declares a trump suite, every other player takes turn to decide if they want to override.
        self.is_initial_game = dealer_position is None # Whether overriding the declaration lets the overrider becomes the dealer.
        self.dealer_position = dealer_position # Position of the dealer. At the start of game 1, the dealer is not determined yet.
        self.defender_points = 0 # Multiples of 5
        
        # Chaodi mode only
        self.enable_chaodi = enable_chaodi
        self.current_chaodi_turn: AbsolutePosition = None

        # Finally, we kick off the game by letting one of the players be the first to draw a card from the deck.
        self.hands[AbsolutePosition.random()].add_card(next(self.deck))

    def get_observation(self, position: AbsolutePosition) -> Observation:
        "Derive the current observation for a given player."
        
        return Observation(
            hand = self.hands[position],
            draw_completed = self.draw_stage_completed,
            dominant_rank = self.dominant_rank,
            declaration = self.declaration.relative_to(position) if self.declaration else None,
            next_declaration_turn = self.current_declaration_turn.relative_to(position) if self.current_declaration_turn else None,
            dealer_position = self.dealer_position.relative_to(position) if self.dealer_position else None,
            defender_points = self.defender_points,
            round_history = [(p.relative_to(position), cards) for p, cards in self.round_history],
            leads = not self.round_history and position == self.dealer_position or self.round_history[-1][0] == position,
            kitty = self.kitty if self.declaration and position == self.declaration.absolute_position else None,
            is_chaodi_turn = self.current_chaodi_turn == position 
        )

    def run_action(self, action: Action, player_position: AbsolutePosition):
        "Run an action on the game, and return the position of the player that needs to act next."
        if isinstance(action, DontDeclareAction):
            if self.current_declaration_turn == player_position: # It's the current player's turn to declare if he wants to. But he chooses not, so the opportunity is passed on to the next player. But if the next player is the one who made the original declaration, then the declaration round is over and we continue drawing cards.
                if player_position.next_position == self.dealer_position:
                    self.current_declaration_turn = None
                    return player_position.next_position
                else:
                    self.current_declaration_turn = self.current_declaration_turn.next_position
                    return self.current_declaration_turn
            elif sum([h.size for h in self.hands.values()]) < 100: # Distribute the next card from the deck if less than 100 are distributed.
                assert self.hands[player_position.next_position].size < 25, "Current player already has 25 cards"
                self.hands[player_position.next_position].add_card(next(self.deck))
                return player_position.next_position
            else: # Otherwise, all 100 cards are drawn from the deck.
                self.draw_stage_completed = True
                if self.dealer_position is None:
                    self.dealer_position = AbsolutePosition.random() # If no one wants to be the dealer, one player is randomly selected.
                for remaining_card in self.deck:
                    self.hands[self.dealer_position].add_card(remaining_card) # Last 8 cards go to the dealer.
                return self.dealer_position
        
        elif isinstance(action, DeclareAction):
            if self.dealer_position is None or self.is_initial_game:
                self.dealer_position = player_position # Round 1, player becomes dealer (抢庄)
            assert self.declaration is None or self.declaration.level < action.declaration.level, "New trump suite declaration must have higher level than the existing one."
            assert self.hands[player_position].get_count(action.declaration.suite, self.dominant_rank) >= 1, "Invalid declaration"
            self.declaration = action.declaration
            if action.declaration.level < 3:
                self.current_declaration_turn = player_position.next_position
                return self.current_declaration_turn
            elif sum([h.size for h in self.hands.values()]) == 100: # If double red joker is declared after all cards are drawn, we move straight to kitty phase -- no need to go around the table. 
                for remaining_card in self.deck:
                    self.hands[self.dealer_position].add_card(remaining_card)
                return self.dealer_position
        
        elif isinstance(action, PlaceKittyAction):
            assert self.kitty.size < 8, "Kitty already has 8 cards"
            self.kitty.add_card(action.card)
            self.hands[player_position].remove_card(action.card)

            if self.kitty.size < 8 or not self.enable_chaodi:
                self.kitty_stage_completed = self.kitty.size == 8
                return player_position
            else:
                self.current_chaodi_turn = player_position.next_position
                return player_position.next_position # Begin chaodi
        
        elif isinstance(action, DontChaodiAction):
            if self.current_chaodi_turn.next_position == self.declaration.absolute_position: # We went around the table and no one declared.
                self.current_chaodi_turn = None
                return self.declaration.absolute_position
            else:
                self.current_chaodi_turn = self.current_chaodi_turn.next_position



