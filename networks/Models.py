from turtle import forward
from typing import Union
import torch
import torch.nn as nn
import sys

class MainModel(nn.Module):
    """
    The main model's observation includes:
        - The player's perceived cardsets for all players based on incomplete information (432)
        - The player's position relative to the dealer
        - The current declaration
        - The position of the current declaration
        - Points earned by opponents and points escaped by defenders
        - Unplayed cards of shape (TOTAL_CARDS)
        - Historical moves of shape (H, 436)
        - Cards played in the current round by every other player, and the position of the player that is leading the round
        - Kitty (if the player is the last to place the kitty), shape (TOTAL_CARDS,)
    """
    def __init__(self, use_oracle=False) -> None:
        super().__init__()

        self.use_oracle = use_oracle
        self.lstm = nn.LSTM(436, 256, batch_first=True)
        if use_oracle:
            self.fc1 = nn.Linear(1197 + 256, 768)
        else:
            self.fc1 = nn.Linear(1197 - 3 * 108 + 256, 768)
        self.fc2 = nn.Linear(768, 512)
        self.fc_rest = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.fc_final = nn.Linear(512, 1)
    
    def forward(self, x: torch.Tensor, history: torch.Tensor):
        # history_out, _ = self.lstm(history)
        # history_out = history_out[:, -1, :]
        _, (history_out, _) = self.lstm(history) # experimental \
        
        x = self.fc1(torch.hstack([x, history_out.squeeze(0)]))
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc_rest(x)
        x = self.fc_final(x)
        return x

class ValueModel(nn.Module):
    """
    The value model's observation includes:
        - The player's perceived cardsets for all players based on incomplete information (432)
        - The player's position relative to the dealer
        - The current declaration
        - The position of the current declaration
        - Points earned by opponents and points escaped by defenders
        - Unplayed cards of shape (108)
        - Historical moves of shape (H, 436)
        - Cards played in the current round by every other player, and the position of the player that is leading the round
        - Kitty (if the player is the last to place the kitty), shape (108,)
    """
    def __init__(self) -> None:
        super().__init__()

        self.lstm = nn.LSTM(436, 256, batch_first=True)
        self.fc1 = nn.Linear(1088 - 3 * 108 + 256, 768)
        self.fc2 = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.fc3 = nn.Linear(512, 1)
    
    def forward(self, x: torch.Tensor, history: torch.Tensor):
        if history.shape[1] > 0:
            history_out, _ = self.lstm(history)
            history_out = history_out[:, -1, :]
        else:
            history_out = torch.zeros((x.shape[0], 256))
        x = self.fc1(torch.hstack([x, history_out]))
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x