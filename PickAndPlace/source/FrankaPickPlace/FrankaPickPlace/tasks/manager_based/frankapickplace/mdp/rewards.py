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
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------


def _stage_complete_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return the staged-command completion mask if the command term exposes one."""
    term = env.command_manager.get_term(command_name)
    if hasattr(term, "stage_complete"):
        return term.stage_complete
    return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)


def _object_within_box_xy(
    env: ManagerBasedRLEnv,
    box_inner_size: float = 0.12,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_floor_cfg: SceneEntityCfg = SceneEntityCfg("target_box_floor"),
) -> torch.Tensor:
    """Return whether the object center lies within the four inner box corners in XY."""
    obj: RigidObject = env.scene[object_cfg.name]
    box_floor: RigidObject = env.scene[box_floor_cfg.name]

    object_xy = obj.data.root_pos_w[:, :2]
    box_center_xy = box_floor.data.root_pos_w[:, :2]
    half_inner_span = 0.5 * box_inner_size

    min_corner_xy = box_center_xy - half_inner_span
    max_corner_xy = box_center_xy + half_inner_span

    inside_x = (object_xy[:, 0] >= min_corner_xy[:, 0]) & (object_xy[:, 0] <= max_corner_xy[:, 0])
    inside_y = (object_xy[:, 1] >= min_corner_xy[:, 1]) & (object_xy[:, 1] <= max_corner_xy[:, 1])
    return inside_x & inside_y


def waypoint_hold_bonus(env: ManagerBasedRLEnv, command_name: str = "transport_target") -> torch.Tensor:
    """Reward progress toward satisfying the required waypoint hold duration."""
    term = env.command_manager.get_term(command_name)
    if not hasattr(term, "waypoint_hold_elapsed") or not hasattr(term, "stage_complete"):
        return torch.zeros(env.num_envs, device=env.device)
    hold_progress = torch.clamp(term.waypoint_hold_elapsed / max(term.cfg.hold_duration_s, 1.0e-6), 0.0, 1.0)
    return (~term.stage_complete).float() * hold_progress


def premature_goal_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "box_pose",
    staged_command_name: str = "transport_target",
    std: float = 0.05,
    minimal_height: float = 0.04,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize moving the lifted object near the final box before the waypoint stage is complete."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    lifted = object.data.root_pos_w[:, 2] > minimal_height
    stage_incomplete = (~_stage_complete_mask(env, staged_command_name)).float()
    return stage_incomplete * lifted.float() * (1 - torch.tanh(distance / std))


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def target_stability_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    proximity_std: float = 0.05,
    joint_vel_std: float = 1.0,
    action_rate_std: float = 0.2,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward keeping the robot quiet once the transported object is already near the current target.

    The reward turns on smoothly as the object approaches the active transport target, and within that
    region it prefers low joint velocity and small action-to-action changes.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )

    object_distance = torch.norm(target_pos_w - obj.data.root_pos_w[:, :3], dim=1)
    near_target = 1.0 - torch.tanh(object_distance / max(proximity_std, 1.0e-6))

    joint_speed = torch.sum(torch.square(robot.data.joint_vel), dim=1)
    action_delta = torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )

    quiet_joints = torch.exp(-joint_speed / max(joint_vel_std, 1.0e-6))
    quiet_actions = torch.exp(-action_delta / max(action_rate_std, 1.0e-6))

    return near_target * quiet_joints * quiet_actions


# def drop_object_reward(
#     env: ManagerBasedRLEnv,
#     distance_threshold: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:  
#     """Reward the agent for dropping the object at the target location."""
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # compute the desired position in the world frame
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
#     # distance of the end-effector to the object: (num_envs,)
#     distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
#     # check if within threshold
#     within_threshold = (distance < distance_threshold).float()
#     # gripper opening reward
#     gripper_joints = robot.data.joint_pos[:, -2:]
#     gripper_opened = torch.where(torch.mean(gripper_joints, dim=1) >= 0.04, 1.0, 0.0)
#     return gripper_opened * within_threshold


def grasp_reward(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward closing the gripper when near the object."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    
    # Get distance from EE to object
    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    
    # Only reward gripper closing when close to object
    near_object = (distance < distance_threshold).float()
    
    # Gripper closure: 0.04 is fully open, 0.0 is fully closed
    gripper_joints = robot.data.joint_pos[:, -1]
    gripper_closure = 1.0 - torch.clamp(gripper_joints / 0.04, 0.0, 1.0)
    # gripper_closure = 1.0 - torch.clamp(torch.mean(gripper_joints, dim=1) / 0.04, 0.0, 1.0)
    return near_object * gripper_closure


def grasp_reward_new(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.04,
    target_xy_threshold: float = 0.02,
    target_height_threshold: float = 0.02,
    command_name: str = "box_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward closing the gripper near the object, but turn it off near the final box-center release zone."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    object_pos_w = object.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    near_object = (distance < distance_threshold).float()

    gripper_joints = robot.data.joint_pos[:, -2:]
    gripper_closure = 1.0 - torch.clamp(torch.mean(gripper_joints, dim=1) / 0.04, 0.0, 1.0)

    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )

    xy_distance_to_target = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    z_distance_to_target = torch.abs(object_pos_w[:, 2] - target_pos_w[:, 2])
    in_release_zone = (
        (xy_distance_to_target < target_xy_threshold) & (z_distance_to_target < target_height_threshold)
    )

    return near_object * gripper_closure * (~in_release_zone).float()


def gripper_hold_in_box_penalty(
    env: ManagerBasedRLEnv,
    target_xy_threshold: float = 0.02,
    target_height_threshold: float = 0.02,
    command_name: str = "box_pose",
    staged_command_name: str = "transport_target",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize keeping the gripper closed inside the final release zone after the waypoint stage is done."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    object_pos_w = object.data.root_pos_w[:, :3]
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )

    xy_distance_to_target = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    z_distance_to_target = torch.abs(object_pos_w[:, 2] - target_pos_w[:, 2])
    in_release_zone = (
        (xy_distance_to_target < target_xy_threshold) & (z_distance_to_target < target_height_threshold)
    ).float()
    final_stage_active = _stage_complete_mask(env, staged_command_name).float()

    gripper_joints = robot.data.joint_pos[:, -2:]
    gripper_closure = 1.0 - torch.clamp(torch.mean(gripper_joints, dim=1) / 0.04, 0.0, 1.0)

    return final_stage_active * in_release_zone * gripper_closure


def placement_height_reward(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.08,
    target_height_offset: float = 0.05,
    command_name: str = "drop_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward lowering the object when it's near the target XY position."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    stage_complete = _stage_complete_mask(env, command_name).float()
    
    # Get target position in world frame
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )
    
    object_pos_w = object.data.root_pos_w
    
    # Check if object is near target in XY plane
    xy_distance = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    near_target_xy = (xy_distance < xy_threshold).float()
    
    # Reward being at the right height (target height + small offset)
    target_height = target_pos_w[:, 2] + target_height_offset
    height_error = torch.abs(object_pos_w[:, 2] - target_height)
    height_reward = torch.exp(-height_error / 0.05)  # Sharp peak at correct height
    
    return stage_complete * near_target_xy * height_reward


def release_reward(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.08,
    command_name: str = "drop_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward opening the gripper when object is correctly positioned above target."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    stage_complete = _stage_complete_mask(env, command_name).float()
    
    # Get target position in world frame
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )
    
    object_pos_w = object.data.root_pos_w
    
    # Check if object is well-positioned
    xy_distance = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    height_distance = torch.abs(object_pos_w[:, 2] - target_pos_w[:, 2])
    
    well_positioned = ((xy_distance < xy_threshold) & (height_distance < height_threshold)).float()
    
    # Gripper opening: 0.0 is closed, 0.04 is open
    gripper_joints = robot.data.joint_pos[:, -1]
    gripper_opening = torch.clamp(gripper_joints / 0.04, 0.0, 1.0)
    # gripper_opening = torch.clamp(torch.mean(gripper_joints, dim=1) / 0.04, 0.0, 1.0)
    
    return stage_complete * well_positioned * gripper_opening


def release_reward_new(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.08,
    min_open_fraction: float = 0.70,
    command_name: str = "drop_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward release only when the object is positioned correctly and the gripper is at least 75% open."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    stage_complete = _stage_complete_mask(env, command_name).float()

    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )

    object_pos_w = object.data.root_pos_w
    xy_distance = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    height_distance = torch.abs(object_pos_w[:, 2] - target_pos_w[:, 2])
    well_positioned = ((xy_distance < xy_threshold) & (height_distance < height_threshold)).float()

    gripper_joints = robot.data.joint_pos[:, -2:]
    if env.common_step_counter % 200 == 0:
      print("joint_pos env0:", robot.data.joint_pos[0].detach().cpu())
      print("gripper joints env0:", robot.data.joint_pos[0, -2:].detach().cpu())

    gripper_opening = torch.clamp(torch.mean(gripper_joints, dim=1) / 0.04, 0.0, 1.0)
    opened_enough = (gripper_opening >= min_open_fraction).float()

    return stage_complete * well_positioned * opened_enough * gripper_opening

# -----------------------------------------------------------------------
# Add Grasp Gating Function 
# -----------------------------------------------------------------------

def grasp_enable_away_from_target(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.10,
    z_threshold: float = 0.10,
    command_name: str = "drop_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Enable grasp reward only when the object is still away from the target."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # target position in world frame
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )

    object_pos_w = obj.data.root_pos_w

    xy_distance = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    z_distance = torch.abs(object_pos_w[:, 2] - target_pos_w[:, 2])

    near_target = (xy_distance < xy_threshold) & (z_distance < z_threshold)

    # 1 when far from target, 0 when near target
    return (~near_target).float()

def grasp_reward_gated(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.04,
    target_xy_threshold: float = 0.10,
    target_z_threshold: float = 0.10,
    target_xy_margin: float = 0.02,
    target_z_margin: float = 0.02,
    command_name: str = "drop_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward closing the gripper near the object, but smoothly turn it off in the final release zone.

    The reward has three components:
    1. End-effector proximity to the object (smooth, not binary).
    2. Gripper closure amount.
    3. A smooth gate that stays near 1 while the object is still being transported and goes to 0
       only when the object is sufficiently close to the final target in both XY and Z.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    # object / ee positions
    object_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_object_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)

    # Smooth proximity reward avoids a hard on/off switch around the grasp threshold.
    near_object = 1.0 - torch.tanh(ee_object_distance / max(distance_threshold, 1.0e-6))

    # target position in world frame
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )

    # gate off grasp reward when object is already near target
    xy_distance_to_target = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    z_distance_to_target = torch.abs(object_pos_w[:, 2] - target_pos_w[:, 2])

    # Smoothly turn off grasp reward as the object enters the release zone.
    xy_gate = torch.clamp(
        (xy_distance_to_target - target_xy_threshold) / max(target_xy_margin, 1.0e-6), 0.0, 1.0
    )
    z_gate = torch.clamp(
        (z_distance_to_target - target_z_threshold) / max(target_z_margin, 1.0e-6), 0.0, 1.0
    )
    grasp_gate = torch.maximum(xy_gate, z_gate)

    # gripper closure: 0.04 is fully open, 0.0 is fully closed
    gripper_joints = robot.data.joint_pos[:, -2:]
    gripper_closure = 1.0 - torch.clamp(torch.mean(gripper_joints, dim=1) / 0.04, 0.0, 1.0)

    return near_object * gripper_closure * grasp_gate


def placed_success_reward(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.05,
    z_threshold: float = 0.05,
    linear_vel_threshold: float = 0.05,
    angular_vel_threshold: float = 0.10,
    gripper_open_threshold: float = 0.03,
    command_name: str = "drop_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward successful placement: object near target, stable, and gripper released."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    stage_complete = _stage_complete_mask(env, command_name).float()

    # target position in world frame
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )

    object_pos_w = obj.data.root_pos_w

    # 1) near target
    xy_distance = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    z_distance = torch.abs(object_pos_w[:, 2] - target_pos_w[:, 2])
    near_target = ((xy_distance < xy_threshold) & (z_distance < z_threshold)).float()

    # 2) stable object
    linear_speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    angular_speed = torch.norm(obj.data.root_ang_vel_w, dim=1)
    stable = (
        (linear_speed < linear_vel_threshold)
        & (angular_speed < angular_vel_threshold)
    ).float()

    # 3) released (simple approximation: gripper is open)
    gripper_joints = robot.data.joint_pos[:, -2:]
    gripper_opening = torch.mean(gripper_joints, dim=1)
    released = (gripper_opening > gripper_open_threshold).float()

    return stage_complete * near_target * stable * released


def placed_success_reward_new(
    env: ManagerBasedRLEnv,
    placed_height_offset: float = 0.03,
    z_threshold: float = 0.02,
    linear_vel_threshold: float = 0.05,
    angular_vel_threshold: float = 0.10,
    min_open_fraction: float = 0.75,
    ee_object_distance_threshold: float = 0.06,
    box_inner_size: float = 0.12,
    command_name: str = "transport_target",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    box_floor_cfg: SceneEntityCfg = SceneEntityCfg("target_box_floor"),
) -> torch.Tensor:
    """Reward a true placement only after the object is in the box, stable, released, and separated."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    box_floor: RigidObject = env.scene[box_floor_cfg.name]
    stage_complete = _stage_complete_mask(env, command_name).float()

    object_pos_w = obj.data.root_pos_w[:, :3]
    object_in_box = _object_within_box_xy(
        env, box_inner_size=box_inner_size, object_cfg=object_cfg, box_floor_cfg=box_floor_cfg
    ).float()

    target_height = box_floor.data.root_pos_w[:, 2] + placed_height_offset
    z_distance = torch.abs(object_pos_w[:, 2] - target_height)
    correct_height = (z_distance < z_threshold).float()

    linear_speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    angular_speed = torch.norm(obj.data.root_ang_vel_w, dim=1)
    stable = (
        (linear_speed < linear_vel_threshold)
        & (angular_speed < angular_vel_threshold)
    ).float()

    gripper_joints = robot.data.joint_pos[:, -2:]
    gripper_open_fraction = torch.clamp(gripper_joints / 0.04, 0.0, 1.0)
    released = torch.all(gripper_open_fraction >= min_open_fraction, dim=1).float()

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_object_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    separated = (ee_object_distance > ee_object_distance_threshold).float()

    return stage_complete * object_in_box * correct_height * stable * released * separated

# -----------------------------------------------------------------------
# Separated Staged Rewards
# -----------------------------------------------------------------------


def reward_stage_reach(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Stage 1: Reward for reaching the object."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    # as the robot approaches the object, it should lower its speed
    object_speed = torch.norm(object.data.root_vel_w, dim=1)

    distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * torch.exp(-object_speed)

# def reward_stage_reach_linear(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     # 1. Get EE and Object positions
#     robot: RigidObject = env.scene[robot_cfg.name]
#     # NOTE: Ensure index -1 is your hand/gripper. If unsure, check your asset.
#     ee_pos_w = robot.data.body_pos_w[:, -1, :] 
#     object: RigidObject = env.scene[object_cfg.name]
#     object_pos_w = object.data.root_pos_w

#     # 2. Calculate Euclidean Distance
#     distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    
#     # 3. The Fix: Negative Distance Reward
#     # This has a constant gradient. 1cm closer is ALWAYS +0.01 reward points.
#     # No saturation.
#     return -distance

def reward_stage_lift(
    env: ManagerBasedRLEnv,
    min_height: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Stage 2: Reward ONLY for lifting height."""
    object: RigidObject = env.scene[object_cfg.name]

    return torch.where(
        object.data.root_pos_w[:, 2] > min_height,
        1.0,
        0.0,
    )


def reward_stage_transport(
    env: ManagerBasedRLEnv,
    std: float = 0.2,
    min_height: float = 0.04,
    command_name: str = "drop_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Stage 3: Reward for XY alignment ONLY when lifted."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    is_lifted = object.data.root_pos_w[:, 2] > min_height

    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )
    object_pos_w = object.data.root_pos_w
    d_obj_target = torch.norm(object_pos_w - target_pos_w, dim=1)

    return torch.where(
        is_lifted,
        1 - torch.tanh(d_obj_target / std),
        torch.zeros_like(d_obj_target),
    )


def reward_stage_place(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.03,  # within 3 cm to target
    command_name: str = "drop_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Stage 4: Reward for lowering and releasing. Only active if above target."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )
    object_pos_w = object.data.root_pos_w

    # 1. Check if distance to target XY is within threshold
    d_obj_target = torch.norm(object_pos_w - target_pos_w, dim=1)
    can_drop = (d_obj_target < distance_threshold).float()

    # 2. Gripper Release Reward
    gripper_joints = robot.data.joint_pos[:, -2:]
    gripper_norm = torch.clamp(torch.mean(gripper_joints, dim=1) / 0.04, 0.0, 1.0)

    return gripper_norm * can_drop
