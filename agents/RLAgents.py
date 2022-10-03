import sys
from typing import List, Tuple
import torch


from .Agent import SJAgent

sys.path.append('.')
from env.Actions import Action, ChaodiAction, DeclareAction, DontChaodiAction, DontDeclareAction, FollowAction, LeadAction, PlaceKittyAction
from env.utils import Stage
from env.Observation import Observation
from networks.Models import *

class DeclareAgent(SJAgent):

    def __init__(self, name: str, model: DeclarationModel) -> None:
        super().__init__(name)

        self.model = model

    def act(self, obs: Observation):
        state_tensor = torch.cat([
            obs.hand.tensor, # (108,)
            obs.dealer_position_tensor, # (4,)
            obs.trump_tensor, # (20,)
            obs.declarer_position_tensor, # (4,)
            obs.perceived_trump_cardsets, # (36,)
        ])
        assert state_tensor.shape[0] == 172

        # Action tensor
        best_value = -torch.inf
        best_action = None
        for action in obs.actions:
            assert isinstance(action, DeclareAction) or isinstance(action, DontDeclareAction), "DeclareAgent can only handle declare actions"
            new_value = self.model(torch.cat([state_tensor, action.tensor]).unsqueeze(0))
            if new_value > best_value:
                best_value = new_value
                best_action = action
        return best_action
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        pass
    

class KittyAgent(SJAgent):
    def __init__(self, name: str, model: KittyModel) -> None:
        super().__init__(name)

        self.model = model
    
    def act(self, obs: Observation):
        state_tensor = torch.cat([
            obs.hand.tensor, # (108,)
            obs.dealer_position_tensor, # (4,)
            obs.trump_tensor, # (20,)
            obs.declarer_position_tensor, # (4,)
            obs.perceived_trump_cardsets, # (36,)
        ])
        assert state_tensor.shape[0] == 172

        # Action tensor
        best_value = -torch.inf
        best_action = None
        for action in obs.actions:
            assert isinstance(action, PlaceKittyAction), "KittyAgent can only handle place kitty actions"
            new_value = self.model(state_tensor.unsqueeze(0), action.tensor.unsqueeze(0))
            if new_value > best_value:
                best_value = new_value
                best_action = action
        return best_action
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        pass


class ChaodiAgent(SJAgent):
    def __init__(self, name: str, model: ChaodiModel) -> None:
        super().__init__(name)

        self.model = model
    
    def act(self, obs: Observation):
        state_tensor = torch.cat([
            obs.hand.tensor, # (108,)
            obs.dealer_position_tensor, # (4,)
            obs.trump_tensor, # (20,)
            obs.declarer_position_tensor, # (4,)
            obs.perceived_trump_cardsets, # (36,)
        ])
        assert state_tensor.shape[0] == 172

        # Action tensor
        best_value = -torch.inf
        best_action = None
        for action in obs.actions:
            assert isinstance(action, ChaodiAction) or isinstance(action, DontChaodiAction), "ChaodiAgent can only handle chaodi decisions"
            new_value = self.model(torch.cat([state_tensor, action.tensor]).unsqueeze(0))
            if new_value > best_value:
                best_value = new_value
                best_action = action
        return best_action
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        pass
     
    
class MainAgent(SJAgent):
    def __init__(self, name: str, model: MainModel) -> None:
        super().__init__(name)

        self.model = model
    
    def act(self, obs: Observation):        
        def reward(a):
            x_batch, history_batch, _ = self.prepare_batch_inputs([(obs, a, 0)])
            return self.model(x_batch, history_batch)

        return max(obs.actions, key=reward)
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        x_batch = torch.zeros((len(samples), 1196))
        history_batch = torch.zeros((len(samples), 10, 436)) # Store up to last 10 rounds of history
        gt_rewards = torch.zeros((len(samples), 1))
        for i, (obs, ac, rw) in enumerate(samples):
            historical_moves, current_moves = obs.historical_moves_tensor
            state_tensor = torch.cat([
                obs.perceived_cardsets, # (432,)
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.chaodi_times_tensor, # (4,)
                obs.points_tensor, # (80,)
                obs.unplayed_cards_tensor, # (108,)
                current_moves, # (328,)
                obs.kitty_tensor
            ])
            x_batch[i] = torch.cat([state_tensor, ac.tensor])
            history_batch[i] = historical_moves
            gt_rewards[i] = rw
        
        return x_batch, history_batch, gt_rewards

