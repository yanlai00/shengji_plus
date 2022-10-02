import torch
from torch import nn

class TrumpEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(19, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x