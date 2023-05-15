from collections import deque
import logging
import pickle
import random
import sys
from typing import Deque, List, Tuple
import numpy as np
import os
from env.CardSet import CardSet

from .Agent import SJAgent, StageModule

sys.path.append('.')
from env.Actions import Action, FollowAction, LeadAction
from env.utils import ORDERING_INDEX, Stage, softmax
from env.Observation import Observation
from networks.Models import *

# A generic class that describes a stage module for the DMC agent.
class DMCModule(StageModule):
    def __init__(self, batch_size: int, tau=0.1, dynamic_encoding=True) -> None:
        self.batch_size = batch_size # preferred batch size
        self.tau = tau # soft weight update parameter
        self._model: nn.Module = None # don't set directly
        self._eval_model: nn.Module = None # don't set directly
        self.loss_fn = nn.MSELoss()
        self.train_loss_history: List[float] = []
        self.optimizer: torch.optim.Optimizer = None
        self.dynamic_encoding = dynamic_encoding
    
    # Use this function to load a pretrained model
    def load_model(self, model: nn.Module):
        self._model = model
        self._model.share_memory()
        self._eval_model = pickle.loads(pickle.dumps(model)).to(next(model.parameters()).device)
        self._eval_model.eval().share_memory()
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99, eps=1e-5)

    # Helper function to prepare `Observation` and `Action` objects into tensors
    # Returns a tensor: (batched observation-action pairs, batched rewards)
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        raise NotImplementedError

    # Training function
    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float]]):
        splits = int(len(samples) / self.batch_size)
        for subsamples in np.array_split(np.array(samples, dtype=object), max(1, splits), axis=0):
            *args, rewards = self.prepare_batch_inputs(subsamples)
            pred = self._model(*args)
            loss = self.loss_fn(pred, rewards)
            self.optimizer.zero_grad()
            loss.backward()
            if torch.isnan(loss):
                def init_weights(m):
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight.data)
                self._model.apply(init_weights)
                print(f"Model {self} encountered nan, reset weights to random.")
            else:
                nn.utils.clip_grad_norm_(self._model.parameters(), 80)
                self.optimizer.step()
                self.train_loss_history.append(loss.detach().item())

        for param, target_param in zip(self._model.parameters(), self._eval_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def act(self, obs: Observation, epsilon=None, training=True):
        def reward(a: Action) -> torch.Tensor:
            *state_and_action, _ = self.prepare_batch_inputs([(obs, a, 0)])
            return self._eval_model(*state_and_action).cpu().item()
        
        if epsilon and random.random() < epsilon:
            return random.choice(obs.actions)
            # return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        else:
            # If in verbose mode, log actions and their probabilities in test time
            if not training and logging.getLogger().level == logging.DEBUG:
                logging.debug(f"Probability of actions ({obs.position.value}):")
                rewards = list(map(reward, obs.actions))
                exp_total = np.sum(np.exp(rewards))
                sorted_actions = sorted(zip(obs.actions, rewards), key=lambda x: x[1], reverse=True)
                for i, (action, rw) in enumerate(sorted_actions):
                    logging.debug(f"{i:2}. {action} (reward={round(rw, 4)}, prob={np.exp(rw) / exp_total:.4f})")
                return sorted_actions[0][0]
            else:
                return max(obs.actions, key=reward)

class MainModule(DMCModule):
    def __init__(self, batch_size: int, use_oracle: bool, tau=0.1, dynamic_encoding=True, sac=False) -> None:
        super().__init__(batch_size, tau, dynamic_encoding=dynamic_encoding)

        self.sac = sac
        self.use_oracle = use_oracle
        self.log_alpha = torch.tensor(1.0).log().cuda()
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float, Tuple[Observation, float, float]]], training=True):
        if self.use_oracle:
            x_batch = torch.zeros((len(samples), 1089 + 108)) # additionally provide other players' hands
        else:
            x_batch = torch.zeros((len(samples), 1089 - 2 * 108))
        history_batch = torch.zeros((len(samples), 15, 436)) # Store up to last 15 rounds of history
        gt_rewards = torch.zeros((len(samples), 1))
        next_action_entropy = torch.zeros(len(samples))
        current_action_entropy = torch.zeros(len(samples))

        for i, (obs, ac, rw, aux) in enumerate(samples):
            assert isinstance(ac, LeadAction) or isinstance(ac, FollowAction)
            historical_moves, current_moves = obs.historical_moves_dynamic_tensor if self.dynamic_encoding else obs.historical_moves_tensor
            cardset = ac.cardset
            if aux is not None:
                (next_obs, current_entropy, next_entropy) = aux
                if current_entropy is not None:
                    current_action_entropy[i] = current_entropy
                if next_entropy is not None:
                    next_action_entropy[i] = next_entropy
            else:
                next_obs, current_entropy, next_entropy = None, None, None
            
            state_tensor = torch.cat([
                obs.dynamic_hand_tensor if self.dynamic_encoding else obs.hand.tensor, # (108,),
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.points_tensor, # (80,)
                obs.unplayed_cards_dynamic_tensor if self.dynamic_encoding else obs.unplayed_cards_tensor, # (108,)
                current_moves, # (328,)
                # obs.current_dominating_player_index, # (3,)
                obs.dominates_all_tensor(cardset), # (1,)
            ])

            if self.use_oracle and training:
                state_tensor = torch.cat([obs.oracle_cardsets, state_tensor])
            elif self.use_oracle:
                state_tensor = torch.cat([torch.zeros(108 * 3), state_tensor])

            if self.dynamic_encoding:
                x_batch[i] = torch.cat([state_tensor, ac.dynamic_tensor(obs.dominant_suit)])
            else:
                x_batch[i] = torch.cat([state_tensor, ac.tensor])
            history_batch[i] = historical_moves
            gt_rewards[i] = rw
        device = next(self._model.parameters()).device
        return x_batch.to(device), history_batch.to(device), current_action_entropy.to(device), next_action_entropy.to(device), gt_rewards.to(device)

    def act(self, obs: Observation, epsilon=None, training=True):
        def reward(a: Action) -> torch.Tensor:
            x_batch, history_batch, *_ = self.prepare_batch_inputs([(obs, a, 0, None)])
            return self._eval_model(x_batch, history_batch).cpu().item()
        rewards = list(map(reward, obs.actions))
        action_distribution = np.exp(rewards) / np.sum(np.exp(rewards))
        entropy = -np.mean(action_distribution * np.log2(action_distribution))
        optimal_index = np.argmax(rewards)
        if epsilon and random.random() < epsilon:
            # If in verbose mode, log actions and their probabilities in test time
            if not training and logging.getLogger().level == logging.DEBUG:
                logging.debug("Probability of actions:")
                sorted_actions = sorted(zip(obs.actions, rewards), key=lambda x: x[1], reverse=True)
                for i, (action, rw) in enumerate(sorted_actions):
                    logging.debug(f"{i:2}. {action} (reward={round(rw, 4)})")
                return sorted_actions[0], action_distribution[optimal_index], entropy
            else:
                return obs.actions[optimal_index], action_distribution[optimal_index], entropy
        else:
            return obs.actions[optimal_index], action_distribution[optimal_index], entropy
    
    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float, Tuple[Observation, float, float]]]):
        splits = int(len(samples) / self.batch_size)
        for subsamples in np.array_split(np.array(samples, dtype=object), max(1, splits), axis=0):
            *args, current_action_entropy, next_action_entropy, rewards = self.prepare_batch_inputs(subsamples)
            pred = self._model(*args)
            loss = self.loss_fn(pred, rewards + torch.exp(self.log_alpha) * next_action_entropy.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            if torch.isnan(loss):
                def init_weights(m):
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight.data)
                self._model.apply(init_weights)
                print(f"Model {self} encountered nan, reset weights to random.")
            else:
                nn.utils.clip_grad_norm_(self._model.parameters(), 80)
                self.optimizer.step()
                self.train_loss_history.append(loss.detach().item())
            
            # Update alpha
            if self.sac:
                alpha_loss = self.log_alpha.exp() * torch.mean(current_action_entropy) + self.log_alpha.exp() * 10
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

        for param, target_param in zip(self._model.parameters(), self._eval_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class DMCAgent(SJAgent):
    def __init__(self, name: str, use_oracle: bool, dynamic_encoding=True, sac=False) -> None:
        super().__init__(name)

        self.main_module: MainModule = MainModule(batch_size=64, use_oracle=use_oracle, dynamic_encoding=dynamic_encoding, sac=sac)
        self.dynamic_encoding = dynamic_encoding
        self.sac = sac
    
    def optimizer_states(self):
        return {
            'main_optim_state': self.main_module.optimizer.state_dict(),
            'alpha_optim_state': self.main_module.alpha_optimizer.state_dict() if self.sac else None,
            'alpha': self.main_module.log_alpha.exp().cpu().item()
        }

    def load_optimizer_states(self, state):
        self.main_module.optimizer.load_state_dict(state['main_optim_state'])
        if self.sac:
            self.main_module.log_alpha.data = torch.tensor(state['alpha']).log()
            self.main_module.alpha_optimizer.load_state_dict(state['alpha_optim_state'])

    def load_models_from_disk(self, train_models):
        # Load models for DMC
        loaded_models = True

        try:
            with open(f'{self.name}/state.pkl', mode='rb') as f:
                state = pickle.load(f)
                self.main_module.use_oracle = state['oracle_duration'] > 0
            with open(f'{self.name}/stats.pkl', mode='rb') as f:
                stats = pickle.load(f)
                iterations = stats[-1]['iterations']
            # If resuming from checkpoint, subtract iterations from oracle duration
            oracle_duration = max(0, state['oracle_duration'] - iterations)
            print(f"Resuming with remaining oracle duration {oracle_duration}")
        except Exception as e:
            loaded_models = False
        main_model: nn.Module = train_models.MainModel(use_oracle=self.main_module.use_oracle).cuda()
        if os.path.exists(f'{self.name}/main.pt'):
            main_model.load_state_dict(torch.load(f'{self.name}/main.pt', map_location='cuda'), strict=False)
            print("Using loaded model for main game")
        else:
            loaded_models = False
        self.main_module.load_model(main_model)
        
        return loaded_models, stats[-1]['iterations'] if loaded_models else 0

    def save_models_to_disk(self):
        torch.save(self.main_module._model.state_dict(), self.name + '/main.pt')
    
    def clear_loss_histories(self):
        self.main_module.train_loss_history.clear()