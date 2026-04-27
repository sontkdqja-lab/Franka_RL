from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def staged_command_final_target(env: ManagerBasedRLEnv, command_name: str = "transport_target") -> torch.Tensor:
    """Return the final drop target from a staged transport command."""
    term = env.command_manager.get_term(command_name)
    if hasattr(term, "final_command"):
        return term.final_command
    return term.command


def staged_command_phase(env: ManagerBasedRLEnv, command_name: str = "transport_target") -> torch.Tensor:
    """Return 0 during waypoint hold and 1 after switching to the final drop stage."""
    term = env.command_manager.get_term(command_name)
    if hasattr(term, "stage_complete"):
        return term.stage_complete.float().unsqueeze(-1)
    return torch.ones((env.num_envs, 1), device=env.device)
