

import argparse
import os
import sys
import torch

from config import cfg
from env.environment import DrivingEnv
from rl.agent import ActorCriticAgent
from rl.train import train, demo
from utils.logger import EpisodeLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="2D Self-Driving Car — Actor-Critic RL"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "demo"],
        default="train",
        help="Run mode: 'train' to train the agent, 'demo' to run a trained model.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable Pygame rendering during training (faster).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to checkpoint directory (default: checkpoints/).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of training episodes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()


    if args.no_render:
        cfg.render_training = False
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.episodes:
        cfg.num_episodes = args.episodes


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")


    env = DrivingEnv(cfg)
    agent = ActorCriticAgent(cfg, device)


    if args.mode == "train":

        if args.resume and os.path.exists(os.path.join(cfg.checkpoint_dir, "actor.pt")):
            agent.load(cfg.checkpoint_dir)
            print("[Main] Resuming from checkpoint.")

        logger = EpisodeLogger()
        try:
            train(cfg, agent, env, logger)
        except KeyboardInterrupt:
            print("\n[Main] Training interrupted. Saving checkpoint...")
            agent.save(cfg.checkpoint_dir)
            from utils.visualization import plot_training_curves
            if logger.num_episodes > 0:
                plot_training_curves(logger, save_dir=cfg.checkpoint_dir)
            print("[Main] Saved. Exiting.")
        finally:
            env.close()

    elif args.mode == "demo":

        ckpt_dir = cfg.checkpoint_dir
        best_dir = os.path.join(ckpt_dir, "best")
        if os.path.exists(os.path.join(best_dir, "actor.pt")):
            agent.load(best_dir)
            print("[Main] Loaded best checkpoint.")
        elif os.path.exists(os.path.join(ckpt_dir, "actor.pt")):
            agent.load(ckpt_dir)
            print("[Main] Loaded latest checkpoint.")
        else:
            print(f"[Main] ERROR: No checkpoint found in {ckpt_dir}/")
            print("       Train the model first:  python main.py --mode train")
            sys.exit(1)

        try:
            demo(cfg, agent, env)
        except KeyboardInterrupt:
            print("\n[Main] Demo interrupted.")
        finally:
            env.close()


if __name__ == "__main__":
    main()
