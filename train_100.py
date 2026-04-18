
import os
os.environ.setdefault("SDL_VIDEODRIVER", "windows")

from config import cfg
from env.environment import DrivingEnv
from rl.agent import ActorCriticAgent
from utils.logger import EpisodeLogger
from rl.train import train
import torch


cfg.num_episodes = 100
cfg.render_training = True
cfg.fps_train = 30
cfg.random_track = True
cfg.log_interval = 10
cfg.checkpoint_interval = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = DrivingEnv(cfg)
agent = ActorCriticAgent(cfg, device)
logger = EpisodeLogger()

train(cfg, agent, env, logger)
env.close()
