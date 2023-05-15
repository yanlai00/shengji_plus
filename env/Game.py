# Describes the current state of a Shengji game. Obeys Markov's property.

from math import ceil
import random
from typing import List, Tuple, Union
from env.Actions import Action, LeadAction, FollowAction

from env.Observation import Observation
from env.utils import *
from .CardSet import CardSet
import logging

class Game:
    def __init__(self, dealer_position: AbsolutePosition = None, deck: List[str] = None, is_warmup_game=False, oracle_value=0.0) -> None:
        # Player information
        self.hands = {
            AbsolutePosition.NORTH: CardSet(),
            AbsolutePosition.SOUTH: CardSet(),
            AbsolutePosition.WEST: CardSet(),
            AbsolutePosition.EAST: CardSet()
        }
        self.public_cards = {
            AbsolutePosition.NORTH: CardSet(),
            AbsolutePosition.SOUTH: CardSet(),
            AbsolutePosition.WEST: CardSet(),
            AbsolutePosition.EAST: CardSet()
        }
        self.unplayed_cards, self.card_list = CardSet.new_deck()
        if is_warmup_game:
            self.card_list = CardSet.get_tutorial_deck()
        else:
            self.card_list = deck or self.card_list
        self.deck = iter(self.card_list) # First 100 are drawn, last 8 given to dealer.

        # Public game information
        self.round_history: List[Tuple[AbsolutePosition, List[CardSet]]] = []
        "A list of past rounds, each having the structure (P, [CardSet...]), where P is one of 'N', 'W', 'S', or 'E', and CardSets are in the order they were played (so for instance, element 0 is played by P)."
        
        self.stage = Stage.main_stage
        self.declarations: List[Declaration] = [] # Information about who declared the trump suit, what it is, and its level
        self.current_declaration_turn: AbsolutePosition = None # Once a player declares a trump suit, every other player takes turn to decide if they want to override.
        self.dealer_position = dealer_position # Position of the dealer. At the start of game 1, the dealer is not determined yet.
        self.opponent_points = 0 # Multiples of 5
        self.defender_points = 0
        self.game_ended = False

        self.points_per_round: List[int] = [] # Positive means defenders escaped points; negative means opponents scored points
        self.final_opponent_reward: float = None
        self.final_defender_reward: float = None

        self.draw_order = []
        self.is_warmup_game = is_warmup_game # In a warm up game, we don't allow declarations for fairness
        self.oracle_value = oracle_value # the coefficient for the oracle
        self.consecutive_moves = 0
    @property
    def dominant_suit(self):
        return self.declarations[-1].suit if self.declarations else TrumpSuit.NT

    @property
    def game_started(self):
        return sum([c.size for c in self.hands.values()]) > 0 or self.game_ended
        
    def start_game(self):
        # Finally, we kick off the game by letting one of the players be the first to draw a card from the deck.
        first_player = self.dealer_position or AbsolutePosition.random()
        next_card = next(self.deck)
        self.hands[first_player].add_card(next_card)
        self.draw_order.append((first_player, next_card))
        logging.debug(f"Starting new game, first player is {first_player}, dealer is {self.dealer_position}")
        return first_player

    def get_observation(self, position: AbsolutePosition) -> Observation:
        "Derive the current observation for a given player."

        # Compute actions
        actions: List[Action] = []

        if self.round_history[-1][0] == position:
            for move in self.hands[position].get_leading_moves():
                actions.append(LeadAction(move))
        else:
            raise NotImplementedError()
        assert actions, f"Agent {position} has no action to choose from!"

        observation = Observation(
            hand = self.hands[position].copy(),
            position = position,
            actions = actions,
            stage = self.stage,
            declaration = self.declarations[-1].relative_to(position) if self.declarations else None,
            dealer_position = self.dealer_position.relative_to(position) if self.dealer_position else None,
            defender_points = self.defender_points,
            opponent_points = self.opponent_points,
            round_history = [(p.relative_to(position), cards[:]) for p, cards in self.round_history],
            unplayed_cards = self.unplayed_cards.copy(),
            leads_current_trick = self.round_history[-1][0] == position if self.round_history else position == self.dealer_position,
            perceived_left = self.public_cards[position.last_position].copy(),
            perceived_right = self.public_cards[position.next_position].copy(),
            perceived_opposite = self.public_cards[position.next_position.next_position].copy(),
            actual_left=self.hands[position.last_position].copy(),
            actual_right=self.hands[position.next_position].copy(),
            actual_opposite=self.hands[position.next_position.next_position].copy(),
            oracle_value=self.oracle_value
        )

        return observation

    def run_action(self, action: Action, player_position: AbsolutePosition) -> Tuple[AbsolutePosition, float]:
        "Run an action on the game, and return the position of the player that needs to act next and the reward for the current action."
        if isinstance(action, LeadAction):
            logging.debug(f"Round {len(self.round_history)}: {player_position.value} leads with {action.move}")

            if not self.round_history[-1][1]:
                self.round_history[-1][1].append(action.move.cardset)
            self.hands[player_position].remove_cardset(action.move.cardset)
            self.unplayed_cards.remove_cardset(action.move.cardset)
            for card in action.move.cardset.card_list():
                if self.public_cards[player_position].has_card(card):
                    self.public_cards[player_position].remove_card(card)
            return player_position.next_position, 0
        
        elif isinstance(action, FollowAction):
            logging.debug(f"Round {len(self.round_history)}: {player_position.value} follows with {action.cardset}")
            lead_position, moves = self.round_history[-1]
            self.hands[player_position].remove_cardset(action.cardset)
            moves.append(action.cardset)
            self.unplayed_cards.remove_cardset(action.cardset)
            for card in action.cardset.card_list():
                if self.public_cards[player_position].has_card(card):
                    self.public_cards[player_position].remove_card(card)
            if len(moves) == 4: # round finished, find max player and update points
                winner_index = CardSet.round_winner(moves, self.dominant_suit)
                
                position_array = [lead_position, lead_position.next_position, lead_position.next_position.next_position, lead_position.last_position]
                dealer_index = position_array.index(self.dealer_position)
                new_leading_position = position_array[winner_index]
                declarer_wins_round = winner_index == dealer_index or (winner_index + 2) % 4 == dealer_index
                # breakpoint()
                if declarer_wins_round:
                    self.defender_points += 1
                    self.points_per_round.append(1)
                    logging.debug(f"Defenders scored 1 trick")
                else:
                    self.opponent_points += 1
                    self.points_per_round.append(-1)
                    logging.debug(f"Opponents scored 1 trick")

                
                # Checks if game is finished
                if self.hands[player_position].size == 0:
                    self.game_ended = True
                    logging.info(f"Game ends! Opponent current points: {self.opponent_points}")
                    logging.debug(f"Points per round: {self.points_per_round}")
                    
                    opponent_reward = 0
                    if self.opponent_points >= 80:
                        opponent_reward = 1 + (self.opponent_points - 80) // 40
                    elif self.opponent_points >= 40:
                        opponent_reward = -1
                    elif self.opponent_points > 0:
                        opponent_reward = -2
                    else:
                        opponent_reward = -3
                    
                    self.final_opponent_reward = opponent_reward
                    self.final_defender_reward = -opponent_reward
                    
                    return None, 0 # Don't return rewards yet. They will be calculated in the end
                else:
                    logging.debug(f"Player {new_leading_position.value} wins round {len(self.round_history)}")
                    self.round_history.append((new_leading_position, [])) # start new round
                return new_leading_position, 0
            else:
                return player_position.next_position, 0
        
        raise AssertionError(f"Unidentified action class {type(action)}: {action}")

    def print_status(self):
        print("Hands:")
        print(f" • North ({self.hands['N'].size}):", self.hands['N'])
        print(f" • West ({self.hands['W'].size}):", self.hands['W'])
        print(f" • South ({self.hands['S'].size}):", self.hands['S'])
        print(f" • East ({self.hands['E'].size}):", self.hands['E'])
        print("")

        print("Public cards:")
        print(f" • North ({self.public_cards['N'].size}):", self.public_cards['N'])
        print(f" • West ({self.public_cards['W'].size}):", self.public_cards['W'])
        print(f" • South ({self.public_cards['S'].size}):", self.public_cards['S'])
        print(f" • East ({self.public_cards['E'].size}):", self.public_cards['E'])
        print("")
        print("Game Status:")
        print("Dealer:", self.dealer_position.value)
        print("Declarers:", self.declarations)
        if self.round_history:
            print(f"Round history ({len(self.round_history)} total):")
            for leader, moves in self.round_history:
                print(f"    {leader} leads: {moves}")
        print("Opponents' total points:", self.opponent_points)

    def return_status(self):
        info = {}
        info['dealer'] = self.dealer_position.value
        info['declarer'] = self.declarations
        info['opponents_total_points'] = self.opponent_points

        rounds = []
        leaders = []
        moves = []
        if self.round_history:
            cnt = 1
            for leader, move in self.round_history:
                rounds.append(cnt)
                leaders.append(leader)
                moves.append(move)
                cnt += 1

        return info, rounds, leaders, moves