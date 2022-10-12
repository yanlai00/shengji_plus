# This model attempts to construct a compact encoding of the game state (which is ~1200 bits long) using a short 16 bit hash. The hash is then used for the purpose of count-based exploration and to encourage the Deep Agent to visit novel states.

import torch
import torch.nn as nn

class StateAutoEncoder(nn.Module):
    def __init__(self, state_dimension: int, hash_length=16, enabled=False) -> None:
        super().__init__()
        self.hash_length = hash_length
        self.state_dimension = state_dimension
        self.enabled = torch.tensor(enabled)
        self.hash_lock = torch.multiprocessing.get_context('spawn').Lock()
        
        # Hash count exploration
        self.hash_counts = torch.zeros(1 << hash_length) # counts for 16 bit combinations
        self.hash_total = torch.tensor(0, dtype=torch.int64) # Only tensors are shared across processes

        self.encoder = nn.Sequential(
            nn.Linear(state_dimension, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, hash_length),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hash_length, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, state_dimension),
            nn.Sigmoid()
        )

        self.loss = nn.MSELoss()
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.005, momentum=0.9)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, state: torch.Tensor):
        encoding = self.encoder(state)
        recovered = self.decoder(encoding)
        return recovered
    
    def update(self, states: torch.Tensor):
        "Fits the provided batch of states, and return the loss."
        jitter = (torch.rand(states.shape) - 0.5) / 2
        pred = self.forward(states + jitter.to(next(self.parameters()).device))
        loss = self.loss(pred, states)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.detach().item()

    
    def _calculate_hash(self, state: torch.Tensor):
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            encoding = self.encoder(state).squeeze()
            bits = torch.round(encoding)
            index = 0
            for i in range(self.hash_length):
                if bits[i] > 0:
                    index += 1 << i
            return index

    def exploration_bonus(self, state: torch.Tensor):
        if not self.enabled:
            return 0
        
        hash_index = self._calculate_hash(state)

        self.hash_lock.acquire()
        new_count = self.hash_counts[hash_index] + 1
        self.hash_counts[hash_index] = new_count
        self.hash_total += 1
        self.hash_lock.release()

        return 1 / torch.sqrt(new_count)      
    
