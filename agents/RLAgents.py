from asyncio.log import logger
from collections import deque
import random
import sys
from typing import Deque, List, Tuple
import torch
import numpy as np


from .Agent import SJAgent

sys.path.append('.')
from env.Actions import Action, ChaodiAction, DeclareAction, DontChaodiAction, DontDeclareAction, FollowAction, LeadAction, PlaceKittyAction
from env.utils import Stage, softmax
from env.Observation import Observation
from networks.Models import *

class DeepAgent(SJAgent):
    def __init__(self, name: str, model: nn.Module, batch_size=64) -> None:
        super().__init__(name)

        self.model = model
        self.batch_size = batch_size
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99)
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
        self.loss_fn = nn.MSELoss()
        self.train_loss_history: List[float] = []
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        raise NotImplementedError
    
    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float]]):
        splits = int(len(samples) / self.batch_size)
        for subsamples in np.array_split(samples, max(1, splits), axis=0):
            *args, rewards = self.prepare_batch_inputs(subsamples)
            pred = self.model(*args)
            loss = self.loss_fn(pred, rewards)
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), 1000)
            self.optimizer.step()
            self.train_loss_history.append(loss.detach().item())

class DeclareAgent(DeepAgent):
    def __init__(self, name: str, model: DeclarationModel, batch_size=64) -> None:
        super().__init__(name, model, batch_size)
        self.model: DeclarationModel

    def act(self, obs: Observation, explore=False, epsilon=None):
        def reward(a: Action):
            x_batch, _ = self.prepare_batch_inputs([(obs, a, 0)])
            return self.model(x_batch)
        
        if explore:
            return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        elif epsilon and random.random() < epsilon:
            return random.choice(obs.actions)
        else:
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
        device = next(self.model.parameters()).device
        return x_batch.to(device), gt_rewards.to(device)


class KittyAgent(DeepAgent):
    def __init__(self, name: str, model: KittyModel, batch_size=32) -> None:
        super().__init__(name, model, batch_size)
        self.model: KittyModel
    
    def act(self, obs: Observation, explore=False, epsilon=None):
        def reward(a: Action):
            state_batch, action_batch, _ = self.prepare_batch_inputs([(obs, a, 0)])
            return self.model(state_batch, action_batch)
        
        if explore:
            return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        elif epsilon and random.random() < epsilon:
            return random.choice(obs.actions)
        else:
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
        device = next(self.model.parameters()).device
        return state_batch.to(device), action_batch.to(device), gt_rewards.to(device)


class ChaodiAgent(DeepAgent):
    def __init__(self, name: str, model: ChaodiModel, batch_size=16) -> None:
        super().__init__(name, model, batch_size)
        self.model: ChaodiModel

    def act(self, obs: Observation, explore=False, epsilon=None):
        def reward(a: Action):
            x_batch, _ = self.prepare_batch_inputs([(obs, a, 0)])
            return self.model(x_batch)

        if explore:
            return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        elif epsilon and random.random() < epsilon:
            return random.choice(obs.actions)
        else:
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
        device = next(self.model.parameters()).device
        return x_batch.to(device), gt_rewards.to(device)
     
    
class MainAgent(DeepAgent):
    def __init__(self, name: str, model: MainModel, batch_size=64) -> None:
        super().__init__(name, model, batch_size)
        self.model: MainModel
    
    def act(self, obs: Observation, explore=False, epsilon=None):
        def reward(a: Action):
            x_batch, history_batch, _ = self.prepare_batch_inputs([(obs, a, 0)])
            return self.model(x_batch, history_batch)

        if explore:
            return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        elif epsilon and random.random() < epsilon:
            return random.choice(obs.actions)
        else:
            return max(obs.actions, key=reward)
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        x_batch = torch.zeros((len(samples), 1196))
        history_batch = torch.zeros((len(samples), 20, 436)) # Store up to last 15 rounds of history
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
        device = next(self.model.parameters()).device
        return x_batch.to(device), history_batch.to(device), gt_rewards.to(device)

# Q learning
class QLMainAgent(DeepAgent):
    def __init__(self, name: str, model: nn.Module, batch_size=64) -> None:
        super().__init__(name, model, batch_size)

        self.target_network = MainModel()
        self.target_network.load_state_dict(model.state_dict())

    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float, Observation]]):
        splits = int(len(samples) / self.batch_size)
        for subsamples in np.array_split(samples, max(1, splits), axis=0):
            pass
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float, Observation]]):
        x_batch = torch.zeros((len(samples), 1196))
        next_x_batch = torch.zeros((len(samples), 1088))
        history_batch = torch.zeros((len(samples), 15, 436)) # Store up to last 15 rounds of history
        next_history_batch = torch.zeros((len(samples), 15, 436))
        gt_rewards = torch.zeros((len(samples), 1))
        for i, (obs, ac, rw, next_obs) in enumerate(samples):
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
            
            next_historical_moves, next_current_moves = next_obs.historical_moves_tensor
            next_state_tensor = torch.cat([
                next_obs.perceived_cardsets, # (432,)
                next_obs.dealer_position_tensor, # (4,)
                next_obs.trump_tensor, # (20,)
                next_obs.declarer_position_tensor, # (4,)
                next_obs.chaodi_times_tensor, # (4,)
                next_obs.points_tensor, # (80,)
                next_obs.unplayed_cards_tensor, # (108,)
                next_current_moves, # (328,)
                next_obs.kitty_tensor
            ])
            next_x_batch[i] = next_state_tensor
            next_history_batch[i] = next_historical_moves
            gt_rewards[i] = rw
        device = next(self.model.parameters()).device
        return x_batch.to(device), history_batch.to(device), gt_rewards.to(device), next_x_batch.to(device), next_history_batch.to(device)