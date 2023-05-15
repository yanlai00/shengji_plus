from collections import deque
import logging
import pickle
import random
import sys, os
from typing import Deque, List, Tuple, TypeVar, Generic
import numpy as np

from .Agent import SJAgent, StageModule
from .DMCAgent import DMCModule

sys.path.append('.')
from env.Actions import Action, FollowAction, LeadAction
from env.Observation import Observation
from networks.Models import *

class MainModule(DMCModule):
    def __init__(self, batch_size: int, tau=0.1, discount=0.95, sac=False, initial_alpha=1.0) -> None:
        super().__init__(batch_size, tau)

        self.discount = discount
        self.sac = sac
        self.v_model: nn.Module = None # don't set directly
        self.value_loss_history = []
        self.log_alpha = torch.tensor(initial_alpha).log().cuda()
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        
    def load_v_model(self, model: nn.Module):
        self.v_model = model
        self.v_model.share_memory()
        self.value_optimizer = torch.optim.RMSprop(self.v_model.parameters(), lr=0.0001, alpha=0.99, eps=1e-5)

    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float, Tuple[Observation, float, float]]]):
        splits = int(len(samples) / self.batch_size)
        for subsamples in np.array_split(np.array(samples, dtype=object), max(1, splits), axis=0):
            state_and_action, history, rewards, action_probs, next_state, next_history, next_action_entropy, terminals = self.prepare_batch_inputs(subsamples)
            # Update Q network
            with torch.no_grad():
                next_values = self.v_model.forward(next_state, next_history)
            
            pred_values = self._model.forward(state_and_action, history)
            target_values = rewards + self.discount * (1 - terminals) * next_values
            q_loss = self.loss_fn(pred_values, target_values)

            self.optimizer.zero_grad()
            q_loss.backward()
            self.optimizer.step()

            # Update Value network
            pred_values = self.v_model(state_and_action[:, :-109], history)
            if self.sac:
                with torch.no_grad():
                    target_values = rewards + self.discount * (1 - terminals) * (next_values + torch.exp(self.log_alpha) * next_action_entropy.unsqueeze(1))
            v_loss = self.loss_fn(pred_values, target_values)

            self.value_optimizer.zero_grad()
            v_loss.backward()
            self.optimizer.step()

            # Update alpha
            if self.sac:
                alpha_loss = torch.mean(-self.log_alpha.exp() * action_probs.log()) + self.log_alpha.exp() * 20
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            self.value_loss_history.append(v_loss.detach().item())
            self.train_loss_history.append(q_loss.detach().item())

        for param, target_param in zip(self._model.parameters(), self._eval_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float, Tuple[Observation, float, float]]]):
        x_batch = torch.zeros((len(samples), 1089 - 2 * 108))
        next_state_batch = torch.zeros((len(samples), 1088 - 3 * 108))
        history_batch = torch.zeros((len(samples), 15, 436)) # Store up to last 15 rounds of history
        next_history_batch = torch.zeros((len(samples), 15, 436))
        gt_rewards = torch.zeros((len(samples), 1))
        action_probs = torch.zeros(len(samples))
        next_action_entropy = torch.zeros(len(samples))
        terminals = torch.zeros(len(samples), 1)
        for i, (obs, ac, rw, aux) in enumerate(samples):
            assert isinstance(ac, LeadAction) or isinstance(ac, FollowAction)
            if aux is not None:
                (next_obs, max_action_prob, next_entropy) = aux
            else:
                next_obs, max_action_prob, next_entropy = None, None, None
            historical_moves, current_moves = obs.historical_moves_dynamic_tensor
            state_tensor = torch.cat([
                obs.dynamic_hand_tensor, # (108,)
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.points_tensor, # (80,)
                obs.unplayed_cards_dynamic_tensor, # (108,)
                current_moves, # (328,)
                obs.dominates_all_tensor(ac.cardset),
            ])
            x_batch[i] = torch.cat([state_tensor, ac.dynamic_tensor(obs.dominant_suit)])
            history_batch[i] = historical_moves
            
            if next_obs is not None:
                next_historical_moves, next_current_moves = next_obs.historical_moves_dynamic_tensor
                next_state_tensor = torch.cat([
                    # next_obs.perceived_cardsets, # (432,)
                    next_obs.dynamic_hand_tensor, # (108,)
                    next_obs.dealer_position_tensor, # (4,)
                    next_obs.trump_tensor, # (20,)
                    next_obs.declarer_position_tensor, # (4,)
                    next_obs.points_tensor, # (80,)
                    next_obs.unplayed_cards_dynamic_tensor, # (108,)
                    next_current_moves, # (328,)
                ])
                next_state_batch[i] = next_state_tensor
                next_history_batch[i] = next_historical_moves
                next_action_entropy[i] = next_entropy
            else:
                terminals[i] = 1
            if max_action_prob is not None:
                action_probs[i] = max_action_prob
            gt_rewards[i] = rw
        device = next(self._model.parameters()).device
        return x_batch.to(device), history_batch.to(device), gt_rewards.to(device), action_probs.to(device), next_state_batch.to(device), next_history_batch.to(device), next_action_entropy.to(device), terminals.to(device)
    
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
            

class DQNAgent(SJAgent):
    def __init__(self, name: str, discount=0.95, sac=False) -> None:
        super().__init__(name)

        self.sac = sac
        self.main_module: MainModule = MainModule(batch_size=64, discount=discount, sac=sac)

    def optimizer_states(self):
        return {
            'main_optim_state': self.main_module.optimizer.state_dict(),
            'value_optim_state': self.main_module.value_optimizer.state_dict(),
            'alpha_optim_state': self.main_module.alpha_optimizer.state_dict() if self.sac else None,
            'alpha': self.main_module.log_alpha.exp().cpu().item()
        }

    def load_optimizer_states(self, state):
        self.main_module.optimizer.load_state_dict(state['main_optim_state'])
        self.main_module.value_optimizer.load_state_dict(state['value_optim_state'])
        if self.sac:
            self.main_module.log_alpha.data = torch.tensor(state['alpha']).log()
            self.main_module.alpha_optimizer.load_state_dict(state['alpha_optim_state'])

    def load_models_from_disk(self, train_models):
        loaded_models = True

        try:
            with open(f'{self.name}/stats.pkl', mode='rb') as f:
                stats = pickle.load(f)
                iterations = stats[-1]['iterations']
        except Exception as e:
            loaded_models = False
        main_model: nn.Module = train_models.MainModel(use_oracle=False).cuda()
        if os.path.exists(f'{self.name}/main.pt'):
            main_model.load_state_dict(torch.load(f'{self.name}/main.pt', map_location='cuda'), strict=False)
            print("Using loaded model for main game")
        else:
            loaded_models = False
        self.main_module.load_model(main_model)

        value_model: nn.Module = train_models.ValueModel().cuda()
        if os.path.exists(f'{self.name}/value.pt'):
            value_model.load_state_dict(torch.load(f'{self.name}/value.pt', map_location='cuda'), strict=False)
        else:
            loaded_models = False
        self.main_module.load_v_model(value_model)

        return loaded_models, stats[-1]['iterations'] if loaded_models else 0

    def save_models_to_disk(self):
        torch.save(self.main_module._model.state_dict(), self.name + '/main.pt')
        torch.save(self.main_module.v_model.state_dict(), self.name + '/value.pt')
    
    def clear_loss_histories(self):
        self.main_module.train_loss_history.clear()
        self.main_module.value_loss_history.clear()