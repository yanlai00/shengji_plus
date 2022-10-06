from turtle import forward
from typing import Union
import torch
import torch.nn as nn
import sys

class DeclarationModel(nn.Module):
    "The declaration model's observation includes: player's cards, player's position relative to the dealer, current declaration (rank and suite), position of current declaration, known trump cards in each player's hands."
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(172 + 7, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        """
        Performs the forward pass of declaration reward prediction. The input tensor should have shape (B, 179), where B is the batch dimension. The first 172 values are for the observation, and last 7 are for the action.
        """
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

class KittyModel(nn.Module):
    "The kitty model's observation includes: the player's cards, the player's position relative to the dealer, current declaration, position of current declaration, known trump cards in each player's hand (same as those in DeclarationModel)."
    def __init__(self) -> None:
        super().__init__()

        self.single_card_embedding = nn.Embedding(54, 32)
        self.fc1 = nn.Linear(172 + 32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor, card: torch.Tensor):
        """
        Performs the forward pass of kitty placement reward prediction. The input tensors should have shape (B, 172) and (B,), where B is the batch dimension. The first 172 values are for the observation, and the last B values are the indexes of the card to discard.
        """

        card_embeddings = self.single_card_embedding(card)

        x = self.fc1(torch.hstack([x, card_embeddings]))
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

class ChaodiModel(nn.Module):
    "The chaodi model's observation includes: the player's cards, the player's position relative to the dealer, current declaration, position of current declaration, known trump cards in each player's hand."
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(178, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        """
        Performs the forward pass of chaodi decision reward prediction. The input tensor should have shape (B, 178), where B is the batch dimension. The first 172 values are for the observation, and the last 6 values are the action.
        """
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

class MainModel(nn.Module):
    """
    The main model's observation includes:
        - The player's perceived cardsets for all players based on incomplete information (432)
        - The player's position relative to the dealer
        - The current declaration
        - The position of the current declaration
        - The number of times each player chaodied
        - Points earned by opponents and points escaped by defenders
        - Unplayed cards of shape (108)
        - Historical moves of shape (H, 436)
        - Cards played in the current round by every other player, and the position of the player that is leading the round
        - Kitty (if the player is the last to place the kitty), shape (108,)
    """
    def __init__(self) -> None:
        super().__init__()

        self.lstm = nn.LSTM(436, 256, batch_first=True)
        self.fc1 = nn.Linear(1196 + 256, 768)
        self.fc2 = nn.Linear(768, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor, history: torch.Tensor):
        if history.shape[1] > 0:
            history_out, _ = self.lstm(history)
            history_out = history_out[:, -1, :]
        else:
            history_out = torch.zeros((x.shape[0], 320))
        x = self.fc1(torch.hstack([x, history_out]))
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x