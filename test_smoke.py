
import sys
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import torch
from config import cfg
from env.environment import DrivingEnv
from rl.agent import ActorCriticAgent

cfg.render_training = False
device = torch.device("cpu")
env = DrivingEnv(cfg)
agent = ActorCriticAgent(cfg, device)

obs = env.reset()
print(f"Obs shape: {obs.shape}, obs[:5]: {obs[:5]}")


if cfg.random_track:
    start_poses = []
    track_lengths = []
    for _ in range(3):
        env.reset()
        pose = env.track.get_start_pose()
        start_poses.append(pose)
        track_lengths.append(env.track.total_length)


    all_same_pose = all(
        abs(start_poses[i][0] - start_poses[0][0]) < 1e-3
        for i in range(1, len(start_poses))
    )
    all_same_length = all(
        abs(track_lengths[i] - track_lengths[0]) < 1e-3
        for i in range(1, len(track_lengths))
    )
    assert not all_same_pose or not all_same_length, (
        "Random track: start poses and lengths should vary across resets, "
        f"but got poses={start_poses}, lengths={track_lengths}"
    )
    print(f"Random track OK — lengths across 3 resets: {[f'{l:.1f}' for l in track_lengths]}")
else:
    print("random_track=False — skipping variability check")


obs = env.reset()
obs_list, action_list, reward_list, done_list, value_list = [], [], [], [], []

for i in range(10):
    action_np, raw_action, value = agent.select_action(obs, deterministic=False)
    next_obs, reward, done, info = env.step(action_np)

    obs_list.append(obs)
    action_list.append(raw_action)
    reward_list.append(reward)
    done_list.append(done)
    value_list.append(value)

    if i == 0:
        print(f"Action: {action_np}, Reward: {reward:.3f}, Done: {done}")

    obs = next_obs
    if done:
        obs = env.reset()


a_loss, c_loss = agent.update_trajectory(obs_list, action_list, reward_list, done_list, value_list, obs)
print(f"Actor loss: {a_loss:.4f}, Critic loss: {c_loss:.4f}")


agent.save("checkpoints")
agent.load("checkpoints")
print("Save/Load OK")


action_np, _, _ = agent.select_action(obs, deterministic=True)
print(f"Deterministic action: {action_np}")

print("ALL TESTS PASSED")
