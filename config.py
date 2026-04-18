

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:


    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.02
    grad_clip: float = 0.5


    hidden_size: int = 128
    num_hidden_layers: int = 2


    num_sensors: int = 9
    sensor_length: float = 150.0
    sensor_spread: float = 180.0
    track_width: float = 80.0
    max_episode_steps: int = 3000


    random_track: bool = True
    track_min_radius: float = 150.0
    track_max_radius: float = 280.0
    track_num_waypoints: int = 12


    car_length: float = 20.0
    car_width: float = 10.0
    max_speed: float = 8.0
    acceleration: float = 0.4
    friction: float = 0.05
    turn_rate: float = 0.04
    dt: float = 1.0


    num_episodes: int = 5000
    checkpoint_interval: int = 100
    log_interval: int = 10
    render_training: bool = True
    checkpoint_dir: str = "checkpoints"


    screen_width: int = 1000
    screen_height: int = 700
    fps_train: int = 0
    fps_demo: int = 60


    reward_progress: float = 1.0
    reward_speed_bonus: float = 0.1
    reward_center_penalty: float = 0.3
    reward_steer_penalty: float = 0.05
    reward_collision: float = -10.0
    reward_alive_cost: float = 0.0

    # derived dimensions
    @property
    def obs_dim(self) -> int:

        return self.num_sensors + 4

    @property
    def act_dim(self) -> int:

        return 2


cfg = Config()
