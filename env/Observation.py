# Encapsulates all the information that's available to a player during a game.

from typing import List, Tuple
from env.Actions import Action, DeclareAction, DontDeclareAction, PlaceKittyAction
from env.CardSet import CardSet
from env.utils import AbsolutePosition, Declaration, RelativePosition


class Observation:
    def __init__(self, hand: CardSet, position: AbsolutePosition, actions: List[Action], draw_completed: bool, dominant_rank: int, declaration: Declaration, next_declaration_turn: RelativePosition, dealer_position: RelativePosition, defender_points: int, round_history: List[Tuple[str, Tuple[CardSet]]], leads_current_trick = False, kitty: CardSet = None, is_chaodi_turn = False) -> None:
        self.hand = hand
        self.position = position
        self.actions = actions
        self.draw_completed = draw_completed
        self.dominant_rank = dominant_rank
        self.declaration = declaration
        self.next_declaration_turn = next_declaration_turn
        self.dealer = dealer_position
        self.defender_points = defender_points
        self.round_history = round_history
        self.leads_current_round = leads_current_trick # If the player is going to lead the next trick
        self.kitty = kitty # Only observable to the last person who placed the kitty. In chaodi mode, this might not be the dealer.

        self.is_chaodi_turn = is_chaodi_turn

    def __repr__(self) -> str:
        return f"Observation(hand={self.hand}, declaration={self.declaration}, dealer={self.dealer})"

