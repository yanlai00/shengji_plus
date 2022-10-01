# Encapsulates all the information that's available to a player during a game.

from typing import List, Tuple
from env.Actions import Action, DeclareAction, DontDeclareAction, PlaceKittyAction
from env.CardSet import CardSet
from env.utils import AbsolutePosition, Declaration, RelativePosition, Stage
import torch

class Observation:
    def __init__(self, hand: CardSet, position: AbsolutePosition, actions: List[Action], stage: Stage, dominant_rank: int, declaration: Declaration, next_declaration_turn: RelativePosition, dealer_position: RelativePosition, defender_points: int, opponent_points: int, round_history: List[Tuple[str, Tuple[CardSet]]], unplayed_cards: CardSet, leads_current_trick = False, kitty: CardSet = None, is_chaodi_turn = False) -> None:
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

        self.is_chaodi_turn = is_chaodi_turn

    def __repr__(self) -> str:
        return f"Observation(hand={self.hand}, declaration={self.declaration}, dealer={self.dealer})"
    
    @property
    def cards_tensor(self):
        return self.hand.tensor

    @property
    def points_tensor(self):
        "Returns a (40, 2) tensor representing the current point situation as observed by the player. First row is for the player's team."
        defenders_points_tensor = torch.zeros(40)
        opponents_points_tensor = torch.zeros(40)
        defenders_points_tensor[:(self.defender_points // 5)] = 1
        opponents_points_tensor[:(self.opponent_points // 5)] = 1

        if self.dealer == RelativePosition.SELF or self.dealer == RelativePosition.OPPOSITE:
            return torch.vstack([defenders_points_tensor, opponents_points_tensor])
        else:
            return torch.vstack([opponents_points_tensor, defenders_points_tensor])
    
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
        "If the player buried the kitty, return information about the kitty. Otherwise, return an empty matrix. Shape: (54,2)"
        return self.kitty.tensor if self.kitty is not None else torch.zeros((54, 2))
    
    @property
    def unplayed_cards_tensor(self):
        return self.unplayed_cards.tensor