from asyncio.log import logger
from collections import deque
import sys
from typing import Deque, List, Tuple
import torch


from .Agent import SJAgent

sys.path.append('.')
from env.Actions import Action, ChaodiAction, DeclareAction, DontChaodiAction, DontDeclareAction, FollowAction, LeadAction, PlaceKittyAction
from env.utils import Stage
from env.Observation import Observation
from networks.Models import *

class DeepAgent(SJAgent):
    def __init__(self, name: str, model: nn.Module, batch_size=32) -> None:
        super().__init__(name)

        self.model = model
        self.batch_size = batch_size
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)
        self.loss_fn = nn.MSELoss()
        self.train_loss_history: List[float] = []
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        raise NotImplementedError
    
    def learn_from_samples(self, samples: Deque[Tuple[Observation, Action, float]]):
        step_count = 0
        while len(samples) >= self.batch_size:
            sample_batch = []
            for _ in range(self.batch_size):
                sample_batch.append(samples.popleft())

            *args, rewards = self.prepare_batch_inputs(sample_batch)
            
            pred = self.model(*args)
            loss = self.loss_fn(pred, rewards)
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), 1000)
            self.optimizer.step()
            self.train_loss_history.append(loss.detach().item())
            step_count += 1
        if step_count > 0:
            logger.info(f"{self.name} performed {step_count} optimization steps")
class DeclareAgent(DeepAgent):

    def __init__(self, name: str, model: DeclarationModel) -> None:
        super().__init__(name, model)
        self.model: DeclarationModel

    def act(self, obs: Observation):
        def reward(a: Action):
            x_batch, _ = self.prepare_batch_inputs([(obs, a, 0)])
            return self.model(x_batch)
        
        return max(obs.actions, key=reward)
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        x_batch = torch.zeros((len(samples), 179))
        gt_rewards = torch.zeros((len(samples), 1))
        for i, (obs, ac, rw) in enumerate(samples):
            assert isinstance(ac, DeclareAction) or isinstance(ac, DontDeclareAction), "DeclareAgent can only handle declare actions"
            state_tensor = torch.cat([
                obs.hand.tensor, # (108,)
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.perceived_trump_cardsets, # (36,)
            ])
            x_batch[i] = torch.cat([state_tensor, ac.tensor])
            gt_rewards[i] = rw
        return x_batch, gt_rewards


class KittyAgent(DeepAgent):
    def __init__(self, name: str, model: KittyModel) -> None:
        super().__init__(name, model)
        self.model: KittyModel
    
    def act(self, obs: Observation):
        def reward(a: Action):
            state_batch, action_batch, _ = self.prepare_batch_inputs([(obs, a, 0)])
            return self.model(state_batch, action_batch)
        
        return max(obs.actions, key=reward)
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        state_batch = torch.zeros((len(samples), 172))
        action_batch = torch.zeros(len(samples), dtype=torch.int)
        gt_rewards = torch.zeros((len(samples), 1))
        for i, (obs, ac, rw) in enumerate(samples):
            assert isinstance(ac, PlaceKittyAction), "KittyAgent can only handle place kitty actions"
            state_tensor = torch.cat([
                obs.hand.tensor, # (108,)
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.perceived_trump_cardsets, # (36,)
            ])
            state_batch[i] = state_tensor
            action_batch[i] = ac.tensor
            gt_rewards[i] = rw
        return state_batch, action_batch, gt_rewards


class ChaodiAgent(DeepAgent):
    def __init__(self, name: str, model: ChaodiModel) -> None:
        super().__init__(name, model)
        self.model: ChaodiModel

    def act(self, obs: Observation):
        def reward(a: Action):
            x_batch, _ = self.prepare_batch_inputs([(obs, a, 0)])
            return self.model(x_batch)

        return max(obs.actions, key=reward)
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        x_batch = torch.zeros((len(samples), 178))
        gt_rewards = torch.zeros((len(samples), 1))
        for i, (obs, ac, rw) in enumerate(samples):
            assert isinstance(ac, ChaodiAction) or isinstance(ac, DontChaodiAction), "ChaodiAgent can only handle chaodi decisions"
            state_tensor = torch.cat([
                obs.hand.tensor, # (108,)
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.perceived_trump_cardsets, # (36,)
            ])
            x_batch[i] = torch.cat([state_tensor, ac.tensor])
            gt_rewards[i] = rw
        return x_batch, gt_rewards
     
    
class MainAgent(DeepAgent):
    def __init__(self, name: str, model: MainModel) -> None:
        super().__init__(name, model)
        self.model: MainModel
    
    def act(self, obs: Observation):        
        def reward(a: Action):
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

