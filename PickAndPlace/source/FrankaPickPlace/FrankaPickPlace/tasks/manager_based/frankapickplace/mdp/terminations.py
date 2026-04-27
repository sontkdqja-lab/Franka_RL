# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _stage_complete_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return the staged-command completion mask if the command term exposes one."""
    term = env.command_manager.get_term(command_name)
    if hasattr(term, "stage_complete"):
        return term.stage_complete
    return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return _stage_complete_mask(env, command_name) & (distance < threshold)


def object_placed_success(
    env: ManagerBasedRLEnv,
    command_name: str = "transport_target",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.02,
    linear_vel_threshold: float = 0.05,
    angular_vel_threshold: float = 0.10,
    gripper_open_threshold: float = 0.03,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate only after the object is placed at the final target and released."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b)

    object_pos_w = obj.data.root_pos_w[:, :3]
    xy_distance = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    z_distance = torch.abs(object_pos_w[:, 2] - target_pos_w[:, 2])
    near_target = (xy_distance < xy_threshold) & (z_distance < z_threshold)

    linear_speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    angular_speed = torch.norm(obj.data.root_ang_vel_w, dim=1)
    stable = (linear_speed < linear_vel_threshold) & (angular_speed < angular_vel_threshold)

    gripper_joints = robot.data.joint_pos[:, -2:]
    released = torch.mean(gripper_joints, dim=1) > gripper_open_threshold

    return _stage_complete_mask(env, command_name) & near_target & stable & released
