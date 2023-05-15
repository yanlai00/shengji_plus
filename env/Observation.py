# Encapsulates all the information that's available to a player during a game.

from typing import List, Tuple
from env.Actions import Action
from env.CardSet import CardSet
from env.utils import LETTER_RANK, AbsolutePosition, Declaration, RelativePosition, Stage, TrumpSuit
import torch
from game_consts import TOTAL_CARDS

class Observation:
    def __init__(self, hand: CardSet, position: AbsolutePosition, actions: List[Action], stage: Stage, declaration: Declaration, dealer_position: RelativePosition, defender_points: int, opponent_points: int, round_history: List[Tuple[RelativePosition, List[CardSet]]], unplayed_cards: CardSet, leads_current_trick: bool, perceived_left = CardSet(), perceived_right = CardSet(), perceived_opposite = CardSet(), actual_left = CardSet(), actual_right = CardSet(), actual_opposite = CardSet(), oracle_value=0.0) -> None:
        self.hand = hand
        self.position = position
        self.actions = actions
        self.stage = stage
        self.declaration = declaration
        self.dealer = dealer_position
        self.defender_points = defender_points
        self.opponent_points = opponent_points
        self.round_history = round_history
        self.unplayed_cards = unplayed_cards
        self.leads_current_round = leads_current_trick # If the player is going to lead the next trick
        self.perceived_left = perceived_left
        self.perceived_right = perceived_right
        self.perceived_opposite = perceived_opposite
        self.actual_left = actual_left
        self.actual_right = actual_right
        self.actual_opposite = actual_opposite

        self.historical_rounds = 14 # TODO
        self.oracle_value = oracle_value # Bernoulli variable parameter

    def __repr__(self) -> str:
        return f"Observation({self.position.value}, hand={self.hand})"
    
    @property
    def dominant_suit(self):
        return self.declaration.suit if self.declaration else TrumpSuit.NT

    @property
    def points_tensor(self):
        "Returns a (26,) tensor representing the current point situation as observed by the player. First 13 values is for the player's team, last 13 for the other team."
        defenders_points_tensor = torch.zeros(13)
        opponents_points_tensor = torch.zeros(13)
        defenders_points_tensor[:(self.defender_points)] = 1
        opponents_points_tensor[:(self.opponent_points)] = 1

        if self.dealer == RelativePosition.SELF or self.dealer == RelativePosition.OPPOSITE:
            return torch.cat([defenders_points_tensor, opponents_points_tensor])
        else:
            return torch.cat([opponents_points_tensor, defenders_points_tensor])
    
    @property
    def dealer_position_tensor(self):
        "Returns a (4,) one-hot tensor representing the dealer's position relative to the player."
        pos = torch.zeros(4)

        if not self.dealer:
            return pos
        
        index = [RelativePosition.SELF, RelativePosition.RIGHT, RelativePosition.OPPOSITE, RelativePosition.LEFT].index(self.dealer)
        pos[index] = 1
        return pos
    
    @property
    def declarer_position_tensor(self):
        "Returns a (4,) one-hot tensor representing the location of the declarer relative to self."
        pos = torch.zeros(4)
        if not self.declaration:
            return pos
        index = [RelativePosition.SELF, RelativePosition.RIGHT, RelativePosition.OPPOSITE, RelativePosition.LEFT].index(self.declaration.relative_position)
        pos[index] = 1
        return pos
    
    @property
    def trump_tensor(self):
        "Returns a (20,) tensor representing the current trump suit and trump rank."
        return self.declaration.tensor if self.declaration else torch.zeros(7)
    
    @property
    def dynamic_hand_tensor(self):
        return self.hand.get_dynamic_tensor(self.declaration.suit if self.declaration else TrumpSuit.NT)
    
    @property
    def unplayed_cards_tensor(self):
        "Returns a (TOTAL_CARDS,) tensor representing all cards not played (and not owned by the current player)."
        unplayed = self.unplayed_cards.copy()
        unplayed.remove_cardset(self.hand)
        return unplayed.tensor
    
    @property
    def unplayed_cards_dynamic_tensor(self):
        "Returns a (TOTAL_CARDS,) tensor representing all cards not played (and not owned by the current player)."
        unplayed = self.unplayed_cards.copy()
        unplayed.remove_cardset(self.hand)
        return unplayed.get_dynamic_tensor(self.dominant_suit)
    
    @property
    def perceived_cardsets(self):
        "Returns the cards for each player from the current player's perspective, starting from themselves going anti-clickwise. Shape: (432,)"
        # Note: compatibility issues with legacy models
        return torch.cat([
            self.perceived_right.get_dynamic_tensor(self.dominant_suit),
            self.perceived_opposite.get_dynamic_tensor(self.dominant_suit),
            self.perceived_left.get_dynamic_tensor(self.dominant_suit)
        ])
    
    @property
    def oracle_cardsets(self):
        perfect_info = torch.cat([
            self.actual_right.get_dynamic_tensor(self.dominant_suit),
            self.actual_opposite.get_dynamic_tensor(self.dominant_suit),
            self.actual_left.get_dynamic_tensor(self.dominant_suit)
        ])
        # Mask using Bernoulli random variables
        mask = torch.bernoulli(torch.ones_like(perfect_info) * self.oracle_value)
        return torch.maximum(perfect_info * mask, self.perceived_cardsets)
    
    @property
    def historical_moves_tensor(self):
        "Returns two tensor of shape (20, 436), (328,) representing the historical rounds of the current game."
        history_tensor = torch.zeros((min(15, len(self.round_history)), 4 + 4 * TOTAL_CARDS))
        position_order = [RelativePosition.SELF, RelativePosition.RIGHT, RelativePosition.OPPOSITE, RelativePosition.LEFT]
        for i, (pos, round) in enumerate(self.round_history[-self.historical_rounds - 1:]):
            current_player_index = position_order.index(pos)
            history_tensor[i, current_player_index] = 1
            for cardset in round:
                history_tensor[i, 4 + TOTAL_CARDS * current_player_index : 4 + TOTAL_CARDS * (current_player_index + 1)] = cardset.tensor
                current_player_index = (current_player_index + 1) % 4
        
        padded_history = torch.vstack([
            torch.zeros((15 - history_tensor.shape[0], 436)),
            history_tensor
        ])
        return padded_history, torch.cat([history_tensor[-1][:4], history_tensor[-1][112:]])
    
    @property
    def historical_moves_dynamic_tensor(self):
        "Returns two tensor of shape (20, 436), (328,) representing the historical rounds of the current game."
        history_tensor = torch.zeros((min(15, len(self.round_history)), 4 + 4 * TOTAL_CARDS))
        position_order = [RelativePosition.SELF, RelativePosition.RIGHT, RelativePosition.OPPOSITE, RelativePosition.LEFT]
        for i, (pos, round) in enumerate(self.round_history[-self.historical_rounds - 1:]):
            current_player_index = position_order.index(pos)
            history_tensor[i, current_player_index] = 1
            for cardset in round:
                history_tensor[i, 4 + TOTAL_CARDS * current_player_index : 4 + TOTAL_CARDS * (current_player_index + 1)] = cardset.get_dynamic_tensor(self.dominant_suit)
                current_player_index = (current_player_index + 1) % 4
        
        padded_history = torch.vstack([
            torch.zeros((15 - history_tensor.shape[0], 436)),
            history_tensor
        ])
        return padded_history, torch.cat([history_tensor[-1][:4], history_tensor[-1][112:]])
    
    @property
    def current_dominating_player_index(self):
        encoding = torch.zeros(3) # first 3 represent which players have played. last 3 represent who's the biggest
        if self.round_history[-1][1]:
            winning_index = CardSet.round_winner(self.round_history[-1][1], self.declaration.suit if self.declaration else TrumpSuit.NT)
            encoding[3 - len(self.round_history[-1][1]) + winning_index] = 1
        return encoding

    def dominates_all_tensor(self, cardset: CardSet):
        if CardSet.round_winner(self.round_history[-1][1] + [cardset], self.declaration.suit if self.declaration else TrumpSuit.NT) == len(self.round_history[-1][1]):
            return torch.tensor([1])
        else:
            return torch.tensor([0])

