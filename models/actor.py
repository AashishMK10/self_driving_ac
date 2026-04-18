

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple

LOG_STD_MIN = -2.0
LOG_STD_MAX = 0.5


class Actor(nn.Module):


    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_size, act_dim)


        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Normal:

        features = self.net(obs)
        mu = self.mu_head(features)


        mu = torch.tanh(mu)


        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        return Normal(mu, std)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        dist = self.forward(obs)

        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()

        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        # map the second action from [-1, 1] to throttle in [0, 1]
        action = raw_action.clone()
        action[..., 0] = torch.clamp(action[..., 0], -1.0, 1.0)
        action[..., 1] = torch.clamp((action[..., 1] + 1.0) / 2.0, 0.0, 1.0)

        return action, log_prob

    def evaluate(
        self, obs: torch.Tensor, action_raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        dist = self.forward(obs)
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
