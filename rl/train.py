

import numpy as np

from env.environment import DrivingEnv
from rl.agent import ActorCriticAgent
from utils.logger import EpisodeLogger
from utils.visualization import plot_training_curves


def train(cfg, agent: ActorCriticAgent, env: DrivingEnv, logger: EpisodeLogger) -> None:

    print("=" * 60)
    print("  Actor-Critic Training (Trajectory + GAE)")
    print(f"  Episodes: {cfg.num_episodes}  |  Max steps: {cfg.max_episode_steps}")
    print(f"  Render: {cfg.render_training}  |  Device: {agent.device}")
    print(f"  GAE lambda: {cfg.gae_lambda}  |  Gamma: {cfg.gamma}")
    print("=" * 60)

    running = True
    best_reward = float("-inf")

    for episode in range(1, cfg.num_episodes + 1):
        obs = env.reset()


        obs_list = []
        action_list = []
        reward_list = []
        done_list = []
        value_list = []

        episode_reward = 0.0
        collisions = 0
        steps = 0
        done = False

        while not done:
            action_np, raw_action, value = agent.select_action(obs, deterministic=False)
            next_obs, reward, done, info = env.step(action_np)

            obs_list.append(obs)
            action_list.append(raw_action)
            reward_list.append(reward)
            done_list.append(done)
            value_list.append(value)

            episode_reward += reward
            if info.get("collision", False):
                collisions += 1
            steps += 1
            obs = next_obs


            if cfg.render_training:
                extra = f"Training ep {episode}/{cfg.num_episodes}"
                running = env.render(fps=cfg.fps_train, extra_info=extra)
                if not running:
                    print("\n[Train] Window closed by user. Stopping.")
                    break

        if not running:
            break


        a_loss, c_loss = agent.update_trajectory(
            obs_list, action_list, reward_list, done_list, value_list, obs
        )


        logger.log_episode(episode_reward, a_loss, c_loss, collisions, steps)

        if episode % cfg.log_interval == 0:
            ma = logger.moving_average(window=50)
            print(
                f"Ep {episode:5d} | "
                f"R={episode_reward:8.1f} | "
                f"MA50={ma:8.1f} | "
                f"Steps={steps:4d} | "
                f"Coll={collisions} | "
                f"A_loss={a_loss:.4f} | "
                f"C_loss={c_loss:.4f}"
            )


        if episode % cfg.checkpoint_interval == 0:
            agent.save(cfg.checkpoint_dir)
            print(f"  -> Checkpoint saved to {cfg.checkpoint_dir}/")


        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(cfg.checkpoint_dir + "/best")


    agent.save(cfg.checkpoint_dir)
    print(f"\nTraining complete. Final checkpoint saved to {cfg.checkpoint_dir}/")
    print(f"Best episode reward: {best_reward:.1f}")


    plot_training_curves(logger, save_dir=cfg.checkpoint_dir)
    print(f"Training plots saved to {cfg.checkpoint_dir}/")


def demo(cfg, agent: ActorCriticAgent, env: DrivingEnv) -> None:

    import pygame

    print("=" * 60)
    print("  Demo Mode - Deterministic Policy")
    print("  Press Q or close window to exit.")
    print("=" * 60)

    agent.actor.eval()
    agent.critic.eval()

    running = True
    while running:
        obs = env.reset()
        done = False

        while not done and running:
            action_np, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action_np)

            extra = f"DEMO  |  Reward: {info['total_reward']:.1f}  |  Speed: {env.car.speed:.1f}"
            running = env.render(fps=cfg.fps_demo, extra_info=extra)


            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                running = False

        if running:
            print(f"  Episode done - Total reward: {info['total_reward']:.1f}  Steps: {info['steps']}")

    env.close()
    print("Demo ended.")
