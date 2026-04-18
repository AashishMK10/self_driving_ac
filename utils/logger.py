

import numpy as np
from typing import List


class EpisodeLogger:


    def __init__(self):
        self.rewards: List[float] = []
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.collisions: List[int] = []
        self.steps: List[int] = []

    def log_episode(
        self,
        reward: float,
        actor_loss: float,
        critic_loss: float,
        collisions: int,
        steps: int,
    ) -> None:

        self.rewards.append(reward)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.collisions.append(collisions)
        self.steps.append(steps)

    def moving_average(self, window: int = 50) -> float:

        if len(self.rewards) == 0:
            return 0.0
        recent = self.rewards[-window:]
        return float(np.mean(recent))

    @property
    def num_episodes(self) -> int:
        return len(self.rewards)
