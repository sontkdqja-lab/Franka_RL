from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_root_state_on_grid(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    grid_x: list[float] | tuple[float, ...],
    grid_y: list[float] | tuple[float, ...],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state onto one of the configured XY grid points."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    grid_points = torch.tensor(
        list(product(grid_x, grid_y)),
        device=asset.device,
        dtype=root_states.dtype,
    )
    grid_indices = torch.randint(0, len(grid_points), (len(env_ids),), device=asset.device)
    selected_xy = grid_points[grid_indices]

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    positions[:, 0:2] = env.scene.env_origins[env_ids, 0:2] + selected_xy

    pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["z", "roll", "pitch", "yaw"]]
    pose_ranges = torch.tensor(pose_range_list, device=asset.device, dtype=root_states.dtype)
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 4), device=asset.device
    )

    positions[:, 2] += pose_samples[:, 0]
    orientations_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 1], pose_samples[:, 2], pose_samples[:, 3]
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    velocity_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    velocity_ranges = torch.tensor(velocity_range_list, device=asset.device, dtype=root_states.dtype)
    velocity_samples = math_utils.sample_uniform(
        velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), device=asset.device
    )
    velocities = root_states[:, 7:13] + velocity_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
