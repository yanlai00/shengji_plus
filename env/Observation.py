# Encapsulates all the information that's available to a player during a game.

from typing import List, Tuple
from env.Actions import Action
from env.CardSet import CardSet
from env.utils import AbsolutePosition, Declaration, RelativePosition, Stage
import torch

class Observation:
    def __init__(self, hand: CardSet, position: AbsolutePosition, actions: List[Action], stage: Stage, dominant_rank: int, declaration: Declaration, next_declaration_turn: RelativePosition, dealer_position: RelativePosition, defender_points: int, opponent_points: int, round_history: List[Tuple[RelativePosition, List[CardSet]]], unplayed_cards: CardSet, leads_current_trick: bool, kitty: CardSet = None, is_chaodi_turn = False, perceived_left = CardSet(), perceived_right = CardSet(), perceived_opposite = CardSet()) -> None:
        self.hand = hand
        self.position = position
        self.actions = actions
        self.stage = stage
        self.dominant_rank = dominant_rank
        self.declaration = declaration
        self.next_declaration_turn = next_declaration_turn
        self.dealer = dealer_position
        self.defender_points = defender_points
        self.opponent_points = opponent_points
        self.round_history = round_history
        self.unplayed_cards = unplayed_cards
        self.leads_current_round = leads_current_trick # If the player is going to lead the next trick
        self.kitty = kitty # Only observable to the last person who placed the kitty. In chaodi mode, this might not be the dealer.
        self.perceived_left = perceived_left
        self.perceived_right = perceived_right
        self.perceived_opposite = perceived_opposite

        self.is_chaodi_turn = is_chaodi_turn

    def __repr__(self) -> str:
        return f"Observation(hand={self.hand}, declaration={self.declaration}, dealer={self.dealer})"

    @property
    def points_tensor(self):
        "Returns a (80,) tensor representing the current point situation as observed by the player. First 40 values is for the player's team, last 40 for the other team."
        defenders_points_tensor = torch.zeros(40)
        opponents_points_tensor = torch.zeros(40)
        defenders_points_tensor[:(self.defender_points // 5)] = 1
        opponents_points_tensor[:(self.opponent_points // 5)] = 1

        if self.dealer == RelativePosition.SELF or self.dealer == RelativePosition.OPPOSITE:
            return torch.cat([defenders_points_tensor, opponents_points_tensor])
        else:
            return torch.cat([opponents_points_tensor, defenders_points_tensor])
    
    @property
    def position_tensor(self):
        "Returns a (4,) one-hot tensor representing the player's position."
        pos = torch.zeros(4)

        if not self.dealer:
            return pos
        
        index = [RelativePosition.SELF, RelativePosition.LEFT, RelativePosition.OPPOSITE, RelativePosition.RIGHT].index(self.dealer)
        pos[index] = 1
        return pos
    
    @property
    def trump_tensor(self):
        "Returns a (19,) tensor representing the current trump suite and trump rank."
        rank_tensor = torch.zeros(13)
        rank_tensor[self.dominant_rank - 2] = 1

        return torch.cat([self.declaration.tensor if self.declaration else torch.zeros(7), rank_tensor])

    @property
    def kitty_tensor(self):
        "If the player buried the kitty, return information about the kitty. Otherwise, return an empty matrix. Shape: (108,)"
        return self.kitty.tensor if self.kitty is not None else torch.zeros(108)
    
    @property
    def unplayed_cards_tensor(self):
        return self.unplayed_cards.tensor
    
    @property
    def perceived_cardsets(self):
        "Returns the cards for each player from the current player's perspective, starting from themselves going anti-clickwise. Shape: (432,)"
        return torch.cat([self.hand.tensor, self.perceived_right.tensor, self.perceived_opposite.tensor, self.perceived_left.tensor])
    
    @property
    def historical_moves_tensor(self):
        "Returns a tensor of shape (H, 436) representing the historical tricks of the current game."
        history_tensor = torch.zeros((len(self.round_history), 4 + 4 * 108))
        position_order = [RelativePosition.SELF, RelativePosition.RIGHT, RelativePosition.OPPOSITE, RelativePosition.LEFT]
        for i, (pos, round) in enumerate(self.round_history):
            current_player_index = position_order.index(pos)
            history_tensor[i, current_player_index] = 1

            for cardset in round:
                history_tensor[i, 4 + 108 * current_player_index : 4 + 108 * (current_player_index + 1)] = cardset.tensor
                current_player_index = (current_player_index + 1) % 4
        
        return history_tensor

            

