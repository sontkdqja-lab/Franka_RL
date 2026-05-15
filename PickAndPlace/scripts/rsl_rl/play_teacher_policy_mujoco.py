#!/usr/bin/env python3

"""Run an exported Isaac Lab PPO teacher policy in MuJoCo with a hand-built observation bridge."""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch


DEFAULT_ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
DEFAULT_GRIPPER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
DEFAULT_ARM_ACTUATOR_NAMES = DEFAULT_ARM_JOINT_NAMES
DEFAULT_GRIPPER_ACTUATOR_NAMES = DEFAULT_GRIPPER_JOINT_NAMES
DEFAULT_ARM_OFFSET = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741], dtype=np.float32)
DEFAULT_FULL_QPOS_OFFSET = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04], dtype=np.float32)
DEFAULT_BOX_POSE = np.array([0.58, -0.36, 0.115, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_WAYPOINT_POSE = np.array([0.50, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_ARM_ACTION_SCALE = 0.5
DEFAULT_GRIPPER_OPEN = 0.04
DEFAULT_GRIPPER_CLOSE = 0.0


def parse_name_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=str, required=True, help="Path to exported PPO teacher policy.pt.")
    parser.add_argument("--xml_path", type=str, required=True, help="Path to the MuJoCo XML/MJCF model.")
    parser.add_argument(
        "--robot_root_body",
        type=str,
        default="panda_link0",
        help="MuJoCo body name used as the Franka root frame for object_position.",
    )
    parser.add_argument(
        "--object_body",
        type=str,
        required=True,
        help="MuJoCo body name for the pick-and-place cube/object root.",
    )
    parser.add_argument(
        "--arm_joint_names",
        type=parse_name_list,
        default=DEFAULT_ARM_JOINT_NAMES,
        help="Comma-separated MuJoCo joint names for the 7-DoF Franka arm.",
    )
    parser.add_argument(
        "--gripper_joint_names",
        type=parse_name_list,
        default=DEFAULT_GRIPPER_JOINT_NAMES,
        help="Comma-separated MuJoCo joint names for the gripper fingers.",
    )
    parser.add_argument(
        "--arm_actuator_names",
        type=parse_name_list,
        default=DEFAULT_ARM_ACTUATOR_NAMES,
        help="Comma-separated MuJoCo actuator names for the 7-DoF Franka arm.",
    )
    parser.add_argument(
        "--gripper_actuator_names",
        type=parse_name_list,
        default=DEFAULT_GRIPPER_ACTUATOR_NAMES,
        help="Comma-separated MuJoCo actuator names for the gripper fingers.",
    )
    parser.add_argument(
        "--arm_default_offset",
        type=float,
        nargs=7,
        default=DEFAULT_ARM_OFFSET.tolist(),
        help="Default Franka arm joint positions used as the Isaac Lab action offset.",
    )
    parser.add_argument(
        "--full_qpos_default_offset",
        type=float,
        nargs=9,
        default=DEFAULT_FULL_QPOS_OFFSET.tolist(),
        help="Default 9-DoF Franka qpos used for joint_pos_rel observation.",
    )
    parser.add_argument(
        "--waypoint_pose",
        type=float,
        nargs=7,
        default=DEFAULT_WAYPOINT_POSE.tolist(),
        help="Waypoint command in robot-root frame: x y z qw qx qy qz.",
    )
    parser.add_argument(
        "--box_pose",
        type=float,
        nargs=7,
        default=DEFAULT_BOX_POSE.tolist(),
        help="Final box command in robot-root frame: x y z qw qx qy qz.",
    )
    parser.add_argument("--arm_action_scale", type=float, default=DEFAULT_ARM_ACTION_SCALE)
    parser.add_argument("--gripper_open_pos", type=float, default=DEFAULT_GRIPPER_OPEN)
    parser.add_argument("--gripper_close_pos", type=float, default=DEFAULT_GRIPPER_CLOSE)
    parser.add_argument("--minimal_height", type=float, default=0.04)
    parser.add_argument("--waypoint_xy_threshold", type=float, default=0.03)
    parser.add_argument("--waypoint_z_threshold", type=float, default=0.03)
    parser.add_argument("--waypoint_max_speed", type=float, default=0.05)
    parser.add_argument("--control_rate_hz", type=float, default=20.0)
    parser.add_argument(
        "--use_policy_gripper_during_transport",
        action="store_true",
        help="Use the PPO gripper output during transport instead of forcing the gripper closed.",
    )
    parser.add_argument(
        "--start_mode",
        choices=("idle", "transport", "release"),
        default="idle",
        help="Initial mode before any key press.",
    )
    return parser.parse_args()


def name_to_joint_qpos_addr(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id == -1:
        raise ValueError(f"Joint '{name}' was not found in the MuJoCo model.")
    return int(model.jnt_qposadr[joint_id])


def name_to_joint_dof_addr(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id == -1:
        raise ValueError(f"Joint '{name}' was not found in the MuJoCo model.")
    return int(model.jnt_dofadr[joint_id])


def name_to_actuator_id(model: mujoco.MjModel, name: str) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id == -1:
        raise ValueError(f"Actuator '{name}' was not found in the MuJoCo model.")
    return actuator_id


def name_to_body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id == -1:
        raise ValueError(f"Body '{name}' was not found in the MuJoCo model.")
    return body_id


def quat_conjugate(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.array([quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def rotate_vector_by_quat(quat_wxyz: np.ndarray, vec_xyz: np.ndarray) -> np.ndarray:
    vec_quat = np.array([0.0, vec_xyz[0], vec_xyz[1], vec_xyz[2]], dtype=np.float32)
    rotated = quat_multiply(quat_multiply(quat_wxyz, vec_quat), quat_conjugate(quat_wxyz))
    return rotated[1:]


def world_to_body_position(root_pos_w: np.ndarray, root_quat_wxyz: np.ndarray, pos_w: np.ndarray) -> np.ndarray:
    return rotate_vector_by_quat(quat_conjugate(root_quat_wxyz), pos_w - root_pos_w)


class TransportCommandState:
    def __init__(
        self,
        waypoint_pose: np.ndarray,
        box_pose: np.ndarray,
        minimal_height: float,
        waypoint_xy_threshold: float,
        waypoint_z_threshold: float,
        waypoint_max_speed: float,
    ):
        self.waypoint_pose = waypoint_pose.astype(np.float32)
        self.box_pose = box_pose.astype(np.float32)
        self.minimal_height = minimal_height
        self.waypoint_xy_threshold = waypoint_xy_threshold
        self.waypoint_z_threshold = waypoint_z_threshold
        self.waypoint_max_speed = waypoint_max_speed
        self.stage_complete = False

    @property
    def command(self) -> np.ndarray:
        return self.box_pose if self.stage_complete else self.waypoint_pose

    @property
    def phase(self) -> np.ndarray:
        return np.array([1.0 if self.stage_complete else 0.0], dtype=np.float32)

    def maybe_reset_for_transport(self, object_height: float):
        if object_height <= self.minimal_height:
            self.stage_complete = False

    def update(self, object_pos_b: np.ndarray, object_lin_vel_w: np.ndarray):
        if self.stage_complete:
            return
        xy_distance = np.linalg.norm(object_pos_b[:2] - self.waypoint_pose[:2])
        z_distance = abs(object_pos_b[2] - self.waypoint_pose[2])
        object_speed = np.linalg.norm(object_lin_vel_w)
        holding_pose = (
            object_pos_b[2] > self.minimal_height
            and xy_distance < self.waypoint_xy_threshold
            and z_distance < self.waypoint_z_threshold
            and object_speed < self.waypoint_max_speed
        )
        if holding_pose:
            self.stage_complete = True


class ModeController:
    def __init__(self, start_mode: str):
        self.mode = start_mode

    def key_callback(self, keycode: int):
        if keycode == ord("1"):
            self.mode = "transport"
            print("[INFO] Mode -> transport")
        elif keycode == ord("2"):
            self.mode = "release"
            print("[INFO] Mode -> release")
        elif keycode == ord("0"):
            self.mode = "idle"
            print("[INFO] Mode -> idle")


def build_teacher_observation(
    data: mujoco.MjData,
    robot_root_body_id: int,
    object_body_id: int,
    qpos_addrs: list[int],
    qvel_addrs: list[int],
    qpos_default_offset: np.ndarray,
    transport_state: TransportCommandState,
    last_action: np.ndarray,
) -> np.ndarray:
    robot_root_pos_w = np.asarray(data.xpos[robot_root_body_id], dtype=np.float32)
    robot_root_quat_wxyz = np.asarray(data.xquat[robot_root_body_id], dtype=np.float32)
    object_pos_w = np.asarray(data.xpos[object_body_id], dtype=np.float32)
    object_lin_vel_w = np.asarray(data.cvel[object_body_id][3:], dtype=np.float32)
    object_pos_b = world_to_body_position(robot_root_pos_w, robot_root_quat_wxyz, object_pos_w)
    transport_state.update(object_pos_b, object_lin_vel_w)

    joint_pos = np.asarray(data.qpos[qpos_addrs], dtype=np.float32)
    joint_vel = np.asarray(data.qvel[qvel_addrs], dtype=np.float32)
    joint_pos_rel = joint_pos - qpos_default_offset

    obs_parts = [
        joint_pos_rel,
        joint_vel,
        object_pos_b,
        transport_state.command,
        transport_state.phase,
        last_action,
    ]
    return np.concatenate(obs_parts, axis=0).astype(np.float32)


def main():
    args = parse_args()
    if len(args.arm_joint_names) != 7 or len(args.arm_actuator_names) != 7:
        raise ValueError("Expected exactly 7 Franka arm joints and 7 arm actuators.")
    if len(args.full_qpos_default_offset) != 9:
        raise ValueError("--full_qpos_default_offset must contain 9 values (7 arm + 2 gripper).")

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)
    teacher_policy = torch.jit.load(args.policy, map_location="cpu")
    teacher_policy.eval()

    qpos_addrs = [name_to_joint_qpos_addr(model, name) for name in args.arm_joint_names + args.gripper_joint_names]
    qvel_addrs = [name_to_joint_dof_addr(model, name) for name in args.arm_joint_names + args.gripper_joint_names]
    arm_actuator_ids = [name_to_actuator_id(model, name) for name in args.arm_actuator_names]
    gripper_actuator_ids = [name_to_actuator_id(model, name) for name in args.gripper_actuator_names]
    robot_root_body_id = name_to_body_id(model, args.robot_root_body)
    object_body_id = name_to_body_id(model, args.object_body)

    arm_default_offset = np.asarray(args.arm_default_offset, dtype=np.float32)
    qpos_default_offset = np.asarray(args.full_qpos_default_offset, dtype=np.float32)
    transport_state = TransportCommandState(
        waypoint_pose=np.asarray(args.waypoint_pose, dtype=np.float32),
        box_pose=np.asarray(args.box_pose, dtype=np.float32),
        minimal_height=float(args.minimal_height),
        waypoint_xy_threshold=float(args.waypoint_xy_threshold),
        waypoint_z_threshold=float(args.waypoint_z_threshold),
        waypoint_max_speed=float(args.waypoint_max_speed),
    )
    mode_controller = ModeController(args.start_mode)

    last_action = np.zeros(8, dtype=np.float32)
    last_arm_target = arm_default_offset.copy()
    last_gripper_target = np.full(len(gripper_actuator_ids), args.gripper_open_pos, dtype=np.float32)

    control_dt = 1.0 / args.control_rate_hz
    sim_dt = float(model.opt.timestep)
    steps_per_control = max(1, int(round(control_dt / sim_dt)))
    step_counter = 0

    print("[INFO] Teacher PPO MuJoCo runner ready.")
    print("[INFO] Key 1 -> transport policy")
    print("[INFO] Key 2 -> release gripper")
    print("[INFO] Key 0 -> idle / hold current pose")

    with mujoco.viewer.launch_passive(model, data, key_callback=mode_controller.key_callback) as viewer:
        while viewer.is_running():
            if step_counter % steps_per_control == 0:
                object_height = float(data.xpos[object_body_id][2])
                if mode_controller.mode == "transport":
                    transport_state.maybe_reset_for_transport(object_height)
                    obs = build_teacher_observation(
                        data=data,
                        robot_root_body_id=robot_root_body_id,
                        object_body_id=object_body_id,
                        qpos_addrs=qpos_addrs,
                        qvel_addrs=qvel_addrs,
                        qpos_default_offset=qpos_default_offset,
                        transport_state=transport_state,
                        last_action=last_action,
                    )
                    obs_t = torch.from_numpy(obs).unsqueeze(0)
                    with torch.inference_mode():
                        policy_action = teacher_policy(obs_t).squeeze(0).detach().cpu().numpy().astype(np.float32)

                    arm_action = np.clip(policy_action[:7], -1.0, 1.0)
                    last_arm_target = arm_default_offset + args.arm_action_scale * arm_action

                    if args.use_policy_gripper_during_transport:
                        gripper_open = policy_action[7] >= 0.0
                        gripper_target = args.gripper_open_pos if gripper_open else args.gripper_close_pos
                        last_gripper_target = np.full(len(gripper_actuator_ids), gripper_target, dtype=np.float32)
                        last_action = policy_action
                    else:
                        last_gripper_target = np.full(len(gripper_actuator_ids), args.gripper_close_pos, dtype=np.float32)
                        last_action = np.concatenate([policy_action[:7], np.array([-1.0], dtype=np.float32)])

                elif mode_controller.mode == "release":
                    last_gripper_target = np.full(len(gripper_actuator_ids), args.gripper_open_pos, dtype=np.float32)
                    last_action = np.concatenate([last_action[:7], np.array([1.0], dtype=np.float32)])

                for actuator_id, target in zip(arm_actuator_ids, last_arm_target, strict=True):
                    data.ctrl[actuator_id] = float(target)
                for actuator_id, target in zip(gripper_actuator_ids, last_gripper_target, strict=True):
                    data.ctrl[actuator_id] = float(target)

            mujoco.mj_step(model, data)
            viewer.sync()
            step_counter += 1
            time.sleep(sim_dt)


if __name__ == "__main__":
    main()
