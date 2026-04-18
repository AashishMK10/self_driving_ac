

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.logger import EpisodeLogger


def _moving_avg(data: list, window: int = 50) -> np.ndarray:

    arr = np.array(data, dtype=np.float64)
    if len(arr) < window:
        return np.cumsum(arr) / np.arange(1, len(arr) + 1)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_training_curves(logger: EpisodeLogger, save_dir: str = "checkpoints") -> None:

    os.makedirs(save_dir, exist_ok=True)
    episodes = np.arange(1, logger.num_episodes + 1)


    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, logger.rewards, alpha=0.3, color="steelblue", label="Episode Reward")
    ma = _moving_avg(logger.rewards, window=50)
    offset = len(episodes) - len(ma)
    ax.plot(episodes[offset:], ma, color="navy", linewidth=2, label="Moving Avg (50)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "reward_curve.png"), dpi=150)
    plt.close(fig)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(episodes, logger.actor_losses, alpha=0.4, color="coral")
    ma_a = _moving_avg(logger.actor_losses, window=50)
    offset_a = len(episodes) - len(ma_a)
    ax1.plot(episodes[offset_a:], ma_a, color="darkred", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Actor Loss")
    ax1.set_title("Actor Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, logger.critic_losses, alpha=0.4, color="mediumseagreen")
    ma_c = _moving_avg(logger.critic_losses, window=50)
    offset_c = len(episodes) - len(ma_c)
    ax2.plot(episodes[offset_c:], ma_c, color="darkgreen", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Critic Loss")
    ax2.set_title("Critic Loss")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)

    print(f"[Viz] Saved reward_curve.png and loss_curves.png to {save_dir}/")
