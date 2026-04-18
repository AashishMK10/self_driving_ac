
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
import numpy as np
from PIL import Image

from config import cfg
from env.environment import DrivingEnv
from rl.agent import ActorCriticAgent
import torch


cfg.render_training = True
cfg.random_track = True
cfg.fps_train = 0
cfg.num_episodes = 30
cfg.max_episode_steps = 400

OUT_DIR = "checkpoints/episode_frames"
GIF_PATH = "checkpoints/track_variation.gif"
os.makedirs(OUT_DIR, exist_ok=True)


pygame.init()
screen = pygame.display.set_mode((cfg.screen_width, cfg.screen_height))

device = torch.device("cpu")
env = DrivingEnv(cfg)
agent = ActorCriticAgent(cfg, device)


env.screen = screen
env.clock = pygame.time.Clock()
env.font = pygame.font.SysFont("consolas", 16)
env._render_initialized = True

frames = []

for episode in range(1, cfg.num_episodes + 1):
    obs = env.reset()


    env.render(fps=0, extra_info=f"Episode {episode}/{cfg.num_episodes}")


    raw = pygame.surfarray.array3d(screen)
    img_arr = np.transpose(raw, (1, 0, 2))
    pil_img = Image.fromarray(img_arr.astype(np.uint8))
    frame_path = os.path.join(OUT_DIR, f"ep_{episode:03d}.png")
    pil_img.save(frame_path)
    frames.append(pil_img.copy())
    print(f"  Episode {episode:3d} | track_length={env.track.total_length:.0f} px | frame saved")


    done = False
    steps = 0
    while not done:
        action_np, raw_action, value = agent.select_action(obs, deterministic=False)
        obs, reward, done, info = env.step(action_np)
        steps += 1

pygame.quit()


thumb_w, thumb_h = cfg.screen_width // 2, cfg.screen_height // 2
thumb_frames = [f.resize((thumb_w, thumb_h), Image.LANCZOS) for f in frames]

thumb_frames[0].save(
    GIF_PATH,
    save_all=True,
    append_images=thumb_frames[1:],
    duration=600,
    loop=0,
    optimize=False,
)
print(f"\nGIF saved -> {GIF_PATH}  ({len(frames)} frames, {thumb_w}x{thumb_h})")
