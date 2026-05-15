from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_cylinder_in_gripper(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hand_body_name: str = "panda_hand",
    left_finger_body_name: str = "panda_leftfinger",
    right_finger_body_name: str = "panda_rightfinger",
    gripper_joint_expr: str = "panda_finger_joint.*",
    gripper_joint_pos: float = 0.017,
    object_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.025),
    object_rot_euler_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Reset the robot to its start pose and place the object along the gripper tool axis."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[asset_cfg.name]

    finger_joint_ids, _ = robot.find_joints(gripper_joint_expr)
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
    joint_pos[:, finger_joint_ids] = gripper_joint_pos
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    robot.update(env.physics_dt)

    hand_body_idx = robot.find_bodies(hand_body_name)[0][0]
    left_finger_body_idx = robot.find_bodies(left_finger_body_name)[0][0]
    right_finger_body_idx = robot.find_bodies(right_finger_body_name)[0][0]

    left_finger_pos_w = robot.data.body_pos_w[env_ids, left_finger_body_idx, :].clone()
    right_finger_pos_w = robot.data.body_pos_w[env_ids, right_finger_body_idx, :].clone()
    grasp_pos_w = 0.5 * (left_finger_pos_w + right_finger_pos_w)
    grasp_quat_w = robot.data.body_quat_w[env_ids, hand_body_idx, :].clone()

    local_offset = torch.tensor(object_offset_b, device=env.device, dtype=grasp_pos_w.dtype).repeat(len(env_ids), 1)
    object_pos_w = grasp_pos_w + quat_apply(grasp_quat_w, local_offset)
    object_vel_w = torch.zeros((len(env_ids), 6), device=env.device, dtype=grasp_pos_w.dtype)

    rot_deg = torch.tensor(object_rot_euler_xyz_deg, device=env.device, dtype=grasp_pos_w.dtype)
    rot_rad = torch.deg2rad(rot_deg).repeat(len(env_ids), 1)
    object_quat_offset = quat_from_euler_xyz(rot_rad[:, 0], rot_rad[:, 1], rot_rad[:, 2])
    object_quat_w = quat_mul(grasp_quat_w, object_quat_offset)

    obj.write_root_pose_to_sim(torch.cat([object_pos_w, object_quat_w], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(object_vel_w, env_ids=env_ids)
    obj.update(env.physics_dt)
