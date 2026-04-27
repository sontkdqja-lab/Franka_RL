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


def drop_object_reward(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:  
    """Reward the agent for dropping the object at the target location."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # check if within threshold
    within_threshold = (distance < distance_threshold).float()
    # gripper opening reward
    gripper_joints = robot.data.joint_pos[:, -2:]
    gripper_opened = torch.where(torch.mean(gripper_joints, dim=1) >= 0.04, 1.0, 0.0)
    return gripper_opened * within_threshold


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
    gripper_joints = robot.data.joint_pos[:, -2:]
    gripper_closure = 1.0 - torch.clamp(torch.mean(gripper_joints, dim=1) / 0.04, 0.0, 1.0)
    
    return near_object * gripper_closure


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
    gripper_joints = robot.data.joint_pos[:, -2:]
    gripper_opening = torch.clamp(torch.mean(gripper_joints, dim=1) / 0.04, 0.0, 1.0)
    
    return stage_complete * well_positioned * gripper_opening

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
    command_name: str = "drop_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward closing the gripper near the object, but disable this reward when near the target."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    # object / ee positions
    object_pos_w = obj.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_object_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)

    # original grasp condition: close to object
    near_object = (ee_object_distance < distance_threshold).float()

    # target position in world frame
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )

    # gate off grasp reward when object is already near target
    xy_distance_to_target = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    z_distance_to_target = torch.abs(object_pos_w[:, 2] - target_pos_w[:, 2])

    near_target = (
        (xy_distance_to_target < target_xy_threshold)
        & (z_distance_to_target < target_z_threshold)
    ).float()

    grasp_gate = 1.0 - near_target

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
