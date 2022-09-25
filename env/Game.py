# Describes the current state of a Shengji game. Obeys Markov's property.

from math import ceil
import random
from typing import List, Tuple, Union
from env.Actions import Action, ChaodiAction, DeclareAction, DontChaodiAction, DontDeclareAction, FollowAction, LeadAction, PlaceKittyAction

from env.Observation import Observation
from env.utils import *
from .CardSet import CardSet, MoveType
import logging

class Game:
    def __init__(self, dominant_rank=2, dealer_position: AbsolutePosition = None, enable_chaodi = True, enable_combos = False) -> None:
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
        self.round_history: List[Tuple[AbsolutePosition, List[CardSet]]] = []
        "A list of past rounds, each having the structure (P, [CardSet...]), where P is one of 'N', 'W', 'S', or 'E', and CardSets are in the order they were played (so for instance, element 0 is played by P)."
        
        self.draw_stage_completed = False # True if all 100 cards are drawn and everyone had their chance to declare/redeclare.
        self.kitty_stage_completed = False # True if the kitty is fixed.
        self.dominant_rank = dominant_rank
        self.declarations: List[Declaration] = [] # Information about who declared the trump suite, what it is, and its level
        self.current_declaration_turn: AbsolutePosition = None # Once a player declares a trump suite, every other player takes turn to decide if they want to override.
        self.is_initial_game = dealer_position is None # Whether overriding the declaration lets the overrider becomes the dealer.
        self.dealer_position = dealer_position # Position of the dealer. At the start of game 1, the dealer is not determined yet.
        self.opponent_points = 0 # Multiples of 5
        self.declarer_points = 0
        self.game_ended = False
        self.kitty_multiplier = None
        
        # Chaodi mode
        self.enable_chaodi = enable_chaodi
        self.current_chaodi_turn: AbsolutePosition = None

        # Combo mode
        self.enable_combos = enable_combos

    @property
    def dominant_suite(self):
        return self.declarations[-1].suite if self.declarations else TrumpSuite.XJ

    @property
    def game_started(self):
        return sum([c.size for c in self.hands.values()]) > 0 or self.game_ended
        
    def start_game(self):
        # Finally, we kick off the game by letting one of the players be the first to draw a card from the deck.
        first_player = self.dealer_position or AbsolutePosition.random()
        self.hands[first_player].add_card(next(self.deck))
        return first_player

    def get_observation(self, position: AbsolutePosition) -> Observation:
        "Derive the current observation for a given player."

        # Compute actions
        actions: List[Action] = []

        if not self.draw_stage_completed: # Stage 1: drawing cards phase
            for suite, level in self.hands[position].trump_declaration_options(self.dominant_rank).items():
                if not self.declarations or self.declarations[-1].level < level and (self.declarations[-1].suite == suite or self.declarations[-1].absolute_position != position) and False:
                    actions.append(DeclareAction(Declaration(suite, level, position)))
            actions.append(DontDeclareAction())
        elif self.hands[position].size > 25: # Stage 2: choosing the kitty
            for card, count in self.hands[position]._cards.items():
                if count > 0: actions.append(PlaceKittyAction(card, count))
        elif self.current_chaodi_turn == position:
            # Chaodi
            if self.enable_chaodi:
                for suite, level in self.hands[position].trump_declaration_options(self.dominant_rank).items():
                    if self.declarations and Declaration.chaodi_level(suite, level) > Declaration.chaodi_level(self.declarations[-1].suite, self.declarations[-1].level):
                        actions.append(ChaodiAction(Declaration(suite, level, position)))
                        print(position.value, "can chaodi using", suite.value)
            actions.append(DontChaodiAction())
        elif self.round_history[-1][0] == position:
            # For training purpose, maybe first turn off combos?
            for move in self.hands[position].get_leading_moves(self.dominant_suite, self.dominant_rank, include_combos=self.enable_combos):
                actions.append(LeadAction(move))
        else:
            # Combo is a catch-all type if we don't know the composition of the cardset
            for cardset in self.hands[position].get_matching_moves(MoveType.Combo(self.round_history[-1][1][0]), self.dominant_suite, self.dominant_rank):
                actions.append(FollowAction(cardset))
        
        assert actions, f"Agent {position} has no action to choose from!"

        observation = Observation(
            hand = self.hands[position],
            position = position,
            actions = actions,
            draw_completed = self.draw_stage_completed,
            dominant_rank = self.dominant_rank,
            declaration = self.declarations[-1].relative_to(position) if self.declarations else None,
            next_declaration_turn = self.current_declaration_turn.relative_to(position) if self.current_declaration_turn else None,
            dealer_position = self.dealer_position.relative_to(position) if self.dealer_position else None,
            defender_points = self.opponent_points,
            round_history = [(p.relative_to(position), cards) for p, cards in self.round_history],
            leads_current_trick = self.round_history[-1][0] == position if self.round_history else position == self.dealer_position,
            kitty = self.kitty if self.declarations and position == self.declarations[-1].absolute_position else None,
            is_chaodi_turn = self.current_chaodi_turn == position 
        )

        return observation

    def run_action(self, action: Action, player_position: AbsolutePosition) -> Tuple[AbsolutePosition, float]:
        "Run an action on the game, and return the position of the player that needs to act next and the reward for the current action."
        if isinstance(action, DontDeclareAction):
            if self.current_declaration_turn == player_position: # It's the current player's turn to declare if he wants to. But he chooses not, so the opportunity is passed on to the next player. But if the next player is the one who made the original declaration, then the declaration round is over and we continue drawing cards.
                if player_position.next_position == self.dealer_position:
                    self.current_declaration_turn = None
                    return player_position.next_position, 0
                else:
                    self.current_declaration_turn = self.current_declaration_turn.next_position
                    return self.current_declaration_turn, 0
            elif sum([h.size for h in self.hands.values()]) < 100: # Distribute the next card from the deck if less than 100 are distributed.
                assert self.hands[player_position.next_position].size < 25, "Current player already has 25 cards"
                self.hands[player_position.next_position].add_card(next(self.deck))
                return player_position.next_position, 0
            else: # Otherwise, all 100 cards are drawn from the deck.
                self.draw_stage_completed = True
                if self.dealer_position is None:
                    self.dealer_position = AbsolutePosition.random() # If no one wants to be the dealer, one player is randomly selected.
                for remaining_card in self.deck:
                    self.hands[self.dealer_position].add_card(remaining_card) # Last 8 cards go to the dealer.
                return self.dealer_position, 0
        
        elif isinstance(action, DeclareAction):
            if self.dealer_position is None or self.is_initial_game:
                self.dealer_position = player_position # Round 1, player becomes dealer (抢庄)
            assert not self.declarations or self.declarations[-1].level < action.declaration.level, "New trump suite declaration must have higher level than the existing one."
            assert self.hands[player_position].get_count(action.declaration.suite, self.dominant_rank) >= 1, "Invalid declaration"
            
            self.declarations.append(action.declaration)
            logging.info(f"Player {player_position} declared {action.declaration.suite} x {1 + int(action.declaration.level > 1)}")
            if action.declaration.level < 3:
                self.current_declaration_turn = player_position.next_position
                return self.current_declaration_turn, 0
            elif sum([h.size for h in self.hands.values()]) == 100: # If double red joker is declared after all cards are drawn, we move straight to kitty phase -- no need to go around the table. 
                for remaining_card in self.deck:
                    self.hands[self.dealer_position].add_card(remaining_card)
                return self.dealer_position, 0
            else:
                self.hands[player_position.next_position].add_card(next(self.deck))
                return player_position.next_position, 0
        
        elif isinstance(action, PlaceKittyAction):
            assert self.kitty.size < 8, "Kitty already has 8 cards"
            self.kitty.add_card(action.card)
            self.hands[player_position].remove_card(action.card)
            logging.info(f"Player {player_position} discarded {action.card} to kitty")
            if self.kitty.size == 8:
                if not self.round_history:
                    self.round_history.append((player_position, []))
                logging.debug(f"Hands of all players:")
                logging.debug(f"  North: {self.hands['N']}")
                logging.debug(f"  West: {self.hands['W']}")
                logging.debug(f"  South: {self.hands['S']}")
                logging.debug(f"  East: {self.hands['E']}")
                logging.debug(f"  Kitty: {self.kitty}")
            if self.kitty.size < 8:
                return player_position, 0 # Current player needs to first finish placing kitty
            if not self.enable_chaodi:
                return self.dealer_position, 0
            elif self.declarations and self.declarations[-1].level == 3:
                self.current_chaodi_turn = None
                return self.dealer_position, 0
            else:
                self.current_chaodi_turn = player_position.next_position
                return player_position.next_position, 0
        
        elif isinstance(action, DontChaodiAction):
            logging.debug(f"Player {player_position} chose not to chaodi")
            # Note: chaodi is only an option if no one declares.
            if self.current_chaodi_turn.next_position == self.declarations[-1].absolute_position: # We went around the table and no one declared.
                self.current_chaodi_turn = None
                return self.declarations[-1].absolute_position, 0
            else:
                self.current_chaodi_turn = self.current_chaodi_turn.next_position
                return self.current_chaodi_turn, 0

        elif isinstance(action, ChaodiAction):
            self.declarations.append(action.declaration)
            self.hands[player_position].add_cardset(self.kitty) # Player picks up kitty
            self.kitty.remove_cardset(self.kitty)
            logging.info(f"Player {player_position} chose to chaodi using {action.declaration.suite}")
            if action.declaration.level == 3:
                self.current_chaodi_turn = None
                return player_position, 0
            else:
                return player_position, 0
        elif isinstance(action, LeadAction):
            logging.info(f"Round {len(self.round_history)}: {player_position.value} leads with {action.move}")
            other_player_hands = [self.hands[player_position.next_position], self.hands[player_position.next_position.next_position], self.hands[player_position.next_position.next_position.next_position]]
            is_legal, penalty_move = CardSet.is_legal_combo(action.move, other_player_hands, self.dominant_suite, self.dominant_rank)
            if is_legal:
                self.round_history[-1][1].append(action.move.cardset)
                self.hands[player_position].remove_cardset(action.move.cardset)
            else:
                self.round_history[-1][1].append(penalty_move)
                self.hands[player_position].remove_cardset(penalty_move)
                logging.info(f"Combo move failed. Player {player_position.value} forced to play {penalty_move}")
            return player_position.next_position, 0
        elif isinstance(action, FollowAction):
            logging.info(f"Round {len(self.round_history)}: {player_position.value} follows with {action.cardset}")
            lead_position, moves = self.round_history[-1]
            self.hands[player_position].remove_cardset(action.cardset)
            moves.append(action.cardset)
            if len(moves) == 4: # round finished, find max player and update points
                winner_index = CardSet.round_winner(moves, self.dominant_suite, self.dominant_rank)
                total_points = sum([c.total_points() for c in moves])
                
                dealer_index = 0
                for i in range(winner_index):
                    if lead_position == self.dealer_position: dealer_index = i
                    lead_position = lead_position.next_position
                
                declarer_wins_round = winner_index == dealer_index or (winner_index + 2) % 4 == dealer_index
                if declarer_wins_round:
                    self.declarer_points += total_points
                else:
                    self.opponent_points += total_points
                
                # Checks if game is finished
                if self.hands[player_position].size == 0:
                    self.game_ended = True
                    logging.info(f"Game ends! Opponent current points: {self.opponent_points}")
                    if not declarer_wins_round:
                        multiplier = MoveType.Combo(moves[winner_index]).get_multiplier(self.dominant_suite, self.dominant_rank)
                        self.opponent_points += self.kitty.total_points() * multiplier
                        logging.info(f"Opponents received {self.kitty.total_points()} x {multiplier} from kitty. Final points: {self.opponent_points}")
                    
                    opponent_reward = 0
                    if self.opponent_points >= 80:
                        opponent_reward = 1 + (self.opponent_points - 80) // 40
                    elif self.opponent_points >= 40:
                        opponent_reward = -1
                    elif self.opponent_points > 0:
                        opponent_reward = -2
                    else:
                        opponent_reward = -3
                    
                    if player_position == self.dealer_position or player_position == self.dealer_position.next_position.next_position:
                        return None, -opponent_reward
                    else:
                        return None, opponent_reward
                else:
                    self.round_history.append((lead_position, [])) # start new round
                    logging.info(f"Player {lead_position.value} wins round {len(self.round_history)}")
                return lead_position, 0 # The next person to lead
            else:
                return player_position.next_position, 0
        
        raise AssertionError(f"Unidentified action class {type(action)}")

    def print_status(self):
        print("Hands:")
        print(f" • North ({self.hands['N'].size}):", self.hands['N'])
        print(f" • West ({self.hands['W'].size}):", self.hands['W'])
        print(f" • South ({self.hands['S'].size}):", self.hands['S'])
        print(f" • East ({self.hands['E'].size}):", self.hands['E'])
        print("")
        print("Game Status:")
        print("Dealer:", self.dealer_position.value)
        print("Declarers:", self.declarations)
        if self.round_history:
            print(f"Round history ({len(self.round_history)} total):")
            for leader, moves in self.round_history:
                print(f"    {leader} leads: {moves}")
        print("Kitty:", self.kitty)
        if self.kitty_multiplier:
            print(f"Opponents won the last round. Kitty points: {self.kitty.total_points()} x {self.kitty_multiplier}")
        print("Opponents' total points:", self.opponent_points)