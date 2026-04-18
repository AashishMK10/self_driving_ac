

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from models.actor import Actor
from models.critic import Critic
from rl.utils import clip_grad


class ActorCriticAgent:


    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device


        self.actor = Actor(cfg.obs_dim, cfg.act_dim, cfg.hidden_size).to(device)
        self.critic = Critic(cfg.obs_dim, cfg.hidden_size).to(device)


        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)


        self.critic_loss_fn = nn.MSELoss()


    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float]:

        obs_t = torch.from_numpy(obs).float().to(self.device)

        with torch.no_grad():
            dist = self.actor(obs_t)
            if deterministic:
                raw_action = dist.mean
            else:
                raw_action = dist.sample()

            value = self.critic(obs_t).squeeze()

        # map the sampled policy output to the env action range
        action = raw_action.clone()
        action[0] = torch.clamp(action[0], -1.0, 1.0)
        action[1] = torch.clamp((action[1] + 1.0) / 2.0, 0.0, 1.0)

        return action.cpu().numpy(), raw_action.cpu().numpy(), value.item()


    def update_trajectory(
        self,
        obs_list: list,
        action_list: list,
        reward_list: list,
        done_list: list,
        value_list: list,
        last_obs: np.ndarray,
    ) -> Tuple[float, float]:

        T = len(reward_list)
        if T == 0:
            return 0.0, 0.0


        with torch.no_grad():
            last_obs_t = torch.from_numpy(last_obs).float().to(self.device)
            last_value = self.critic(last_obs_t).squeeze().item()


        rewards = np.array(reward_list, dtype=np.float32)
        values = np.array(value_list, dtype=np.float32)
        dones = np.array(done_list, dtype=np.float32)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        # compute generalized advantage estimates backward through the episode
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values

        # normalize advantages to reduce update variance
        if T > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        obs_t = torch.from_numpy(np.stack(obs_list)).float().to(self.device)
        actions_t = torch.from_numpy(np.stack(action_list)).float().to(self.device)
        returns_t = torch.from_numpy(returns).float().to(self.device)
        advantages_t = torch.from_numpy(advantages).float().to(self.device)


        values_pred = self.critic(obs_t).squeeze(-1)
        critic_loss = self.critic_loss_fn(values_pred, returns_t)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.cfg.grad_clip > 0:
            clip_grad(self.critic, self.cfg.grad_clip)
        self.critic_optimizer.step()


        dist = self.actor(obs_t)
        log_probs = dist.log_prob(actions_t).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        actor_loss = -(log_probs * advantages_t).mean() - self.cfg.entropy_coef * entropy.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.cfg.grad_clip > 0:
            clip_grad(self.actor, self.cfg.grad_clip)
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()


    def save(self, directory: str) -> None:

        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(directory, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(directory, "critic.pt"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(directory, "actor_opt.pt"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(directory, "critic_opt.pt"))

    def load(self, directory: str) -> None:

        actor_path = os.path.join(directory, "actor.pt")
        critic_path = os.path.join(directory, "critic.pt")
        actor_opt_path = os.path.join(directory, "actor_opt.pt")
        critic_opt_path = os.path.join(directory, "critic_opt.pt")

        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device, weights_only=True))
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device, weights_only=True))
        if os.path.exists(actor_opt_path):
            self.actor_optimizer.load_state_dict(torch.load(actor_opt_path, map_location=self.device, weights_only=True))
        if os.path.exists(critic_opt_path):
            self.critic_optimizer.load_state_dict(torch.load(critic_opt_path, map_location=self.device, weights_only=True))

        print(f"[Agent] Loaded checkpoint from {directory}")
