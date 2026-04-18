

import torch
import torch.nn as nn


class Critic(nn.Module):


    def __init__(self, obs_dim: int, hidden_size: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:

        return self.net(obs)
