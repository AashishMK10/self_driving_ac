
import sys
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import torch
from config import cfg
from env.environment import DrivingEnv
from rl.agent import ActorCriticAgent
from rl.train import train
from utils.logger import EpisodeLogger


cfg.num_episodes = 5
cfg.render_training = False
cfg.log_interval = 1
cfg.checkpoint_interval = 5
cfg.max_episode_steps = 100

device = torch.device("cpu")
env = DrivingEnv(cfg)
agent = ActorCriticAgent(cfg, device)
logger = EpisodeLogger()

train(cfg, agent, env, logger)

print(f"\nLogged {logger.num_episodes} episodes")
print(f"Rewards: {[f'{r:.1f}' for r in logger.rewards]}")
print(f"Files in checkpoints/: {os.listdir('checkpoints')}")
print("TRAINING PIPELINE TEST PASSED")
