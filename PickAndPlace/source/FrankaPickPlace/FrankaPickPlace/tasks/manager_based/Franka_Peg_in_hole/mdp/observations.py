from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def peg_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Peg root position expressed in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    peg: RigidObject = env.scene[peg_cfg.name]
    peg_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, peg.data.root_pos_w[:, :3])
    return peg_pos_b


def ee_to_peg_vector(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Vector from end-effector to peg center in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    peg: RigidObject = env.scene[peg_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    peg_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, peg.data.root_pos_w[:, :3])
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_frame.data.target_pos_w[..., 0, :]
    )
    return peg_pos_b - ee_pos_b


def peg_tip_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    peg_half_length: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Peg tip position expressed in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    peg: RigidObject = env.scene[peg_cfg.name]

    local_tip_offset = torch.zeros((env.num_envs, 3), device=peg.data.root_pos_w.device)
    local_tip_offset[:, 2] = -peg_half_length
    peg_tip_w = peg.data.root_pos_w[:, :3] + quat_apply(peg.data.root_quat_w, local_tip_offset)
    peg_tip_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, peg_tip_w)
    return peg_tip_b


def peg_tip_to_hole_vector(
    env: ManagerBasedRLEnv,
    command_name: str,
    peg_half_length: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Vector from peg tip to hole target, expressed in the robot root frame."""
    del robot_cfg, peg_cfg  # kept for a consistent signature
    peg_tip_b = peg_tip_position_in_robot_root_frame(env, peg_half_length=peg_half_length)
    hole_target_b = env.command_manager.get_command(command_name)[:, :3]
    return hole_target_b - peg_tip_b


def peg_upright_projection(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Cosine similarity between peg axis and world up direction."""
    peg: RigidObject = env.scene[peg_cfg.name]
    local_axis = torch.zeros((env.num_envs, 3), device=peg.data.root_pos_w.device)
    local_axis[:, 2] = 1.0
    peg_axis_w = quat_apply(peg.data.root_quat_w, local_axis)
    return peg_axis_w[:, 2:3]
