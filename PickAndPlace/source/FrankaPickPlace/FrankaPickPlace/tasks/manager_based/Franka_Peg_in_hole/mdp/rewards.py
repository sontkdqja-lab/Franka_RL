# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _hole_target_world(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    target_pos_b = env.command_manager.get_command(command_name)[:, :3]
    target_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b)
    return target_pos_w


def _peg_tip_world(
    env: ManagerBasedRLEnv,
    peg_half_length: float,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    peg: RigidObject = env.scene[peg_cfg.name]
    local_tip_offset = torch.zeros((env.num_envs, 3), device=peg.data.root_pos_w.device)
    local_tip_offset[:, 2] = -peg_half_length
    return peg.data.root_pos_w[:, :3] + quat_apply(peg.data.root_quat_w, local_tip_offset)


def _peg_upright_projection(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    peg: RigidObject = env.scene[peg_cfg.name]
    local_axis = torch.zeros((env.num_envs, 3), device=peg.data.root_pos_w.device)
    local_axis[:, 2] = 1.0
    peg_axis_w = quat_apply(peg.data.root_quat_w, local_axis)
    return torch.clamp(peg_axis_w[:, 2], min=0.0, max=1.0)


def _gripper_closed_fraction(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    gripper_joints = robot.data.joint_pos[:, -2:]
    return 1.0 - torch.clamp(torch.mean(gripper_joints, dim=1) / 0.04, 0.0, 1.0)


def peg_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Reward the agent for lifting the peg above the tabletop."""
    peg: RigidObject = env.scene[peg_cfg.name]
    return torch.where(peg.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def peg_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the end-effector for approaching the peg."""
    peg: RigidObject = env.scene[peg_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    distance = torch.norm(peg.data.root_pos_w[:, :3] - ee_frame.data.target_pos_w[..., 0, :], dim=1)
    return 1.0 - torch.tanh(distance / std)


def grasp_peg_reward(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.045,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward closing the gripper while the peg is within grasping distance."""
    peg: RigidObject = env.scene[peg_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    _ = env.scene[robot_cfg.name]

    distance = torch.norm(peg.data.root_pos_w[:, :3] - ee_frame.data.target_pos_w[..., 0, :], dim=1)
    near_peg = (distance < distance_threshold).float()
    return near_peg * _gripper_closed_fraction(env, robot_cfg=robot_cfg)


def peg_upright_reward(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Reward keeping the peg axis aligned with world up."""
    return _peg_upright_projection(env, peg_cfg=peg_cfg)


def peg_upright_lifted_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Reward keeping the peg upright only after it has been lifted."""
    peg: RigidObject = env.scene[peg_cfg.name]
    lifted = peg.data.root_pos_w[:, 2] > minimal_height
    upright = _peg_upright_projection(env, peg_cfg=peg_cfg)
    return lifted.float() * upright


def peg_hole_xy_alignment_reward(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    peg_half_length: float,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Reward XY alignment between the peg tip and the hole center after lifting."""
    peg: RigidObject = env.scene[peg_cfg.name]
    peg_tip_w = _peg_tip_world(env, peg_half_length=peg_half_length, peg_cfg=peg_cfg)
    hole_target_w = _hole_target_world(env, command_name=command_name)
    xy_distance = torch.norm(peg_tip_w[:, :2] - hole_target_w[:, :2], dim=1)
    lifted = peg.data.root_pos_w[:, 2] > minimal_height
    upright = _peg_upright_projection(env, peg_cfg=peg_cfg)
    return lifted.float() * (1.0 - torch.tanh(xy_distance / std)) * upright


def peg_pre_insertion_reward(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    target_height_offset: float,
    command_name: str,
    peg_half_length: float,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward hovering the peg tip slightly above the hole before insertion."""
    del robot_cfg  # kept for a consistent external signature
    peg_tip_w = _peg_tip_world(env, peg_half_length=peg_half_length, peg_cfg=peg_cfg)
    hole_target_w = _hole_target_world(env, command_name=command_name)
    upright = _peg_upright_projection(env, peg_cfg=peg_cfg)
    gripper_closed = _gripper_closed_fraction(env)

    xy_distance = torch.norm(peg_tip_w[:, :2] - hole_target_w[:, :2], dim=1)
    target_tip_height = hole_target_w[:, 2] + target_height_offset
    height_error = torch.abs(peg_tip_w[:, 2] - target_tip_height)

    xy_reward = torch.exp(-xy_distance / xy_threshold)
    height_reward = torch.exp(-height_error / 0.02)
    return xy_reward * height_reward * upright * gripper_closed


def peg_insertion_reward(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    desired_depth: float,
    command_name: str,
    peg_half_length: float,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Reward insertion depth once the peg is aligned over the hole."""
    peg_tip_w = _peg_tip_world(env, peg_half_length=peg_half_length, peg_cfg=peg_cfg)
    hole_target_w = _hole_target_world(env, command_name=command_name)
    upright = _peg_upright_projection(env, peg_cfg=peg_cfg)
    gripper_closed = _gripper_closed_fraction(env)

    xy_distance = torch.norm(peg_tip_w[:, :2] - hole_target_w[:, :2], dim=1)
    depth = torch.clamp(hole_target_w[:, 2] - peg_tip_w[:, 2], min=0.0, max=desired_depth)
    depth_reward = depth / desired_depth
    xy_gate = (xy_distance < xy_threshold).float()
    return xy_gate * depth_reward * upright * gripper_closed


def peg_insertion_success_reward(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    success_depth: float,
    command_name: str,
    peg_half_length: float,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Sparse reward for a successful insertion."""
    peg_tip_w = _peg_tip_world(env, peg_half_length=peg_half_length, peg_cfg=peg_cfg)
    hole_target_w = _hole_target_world(env, command_name=command_name)
    upright = _peg_upright_projection(env, peg_cfg=peg_cfg)

    xy_distance = torch.norm(peg_tip_w[:, :2] - hole_target_w[:, :2], dim=1)
    depth = hole_target_w[:, 2] - peg_tip_w[:, 2]
    success = (xy_distance < xy_threshold) & (depth > success_depth) & (upright > 0.95)
    return success.float()
