#!/usr/bin/env python3

"""Run an image+qpos student policy in MuJoCo with button-controlled transport/release modes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from train_image_qpos_resnet_mlp_student import ImageQposResNetMLPPolicy


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
DEFAULT_ARM_ACTION_SCALE = 0.5
DEFAULT_GRIPPER_OPEN = 0.04
DEFAULT_GRIPPER_CLOSE = 0.0


def parse_name_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--policy", type=str, help="Path to exported TorchScript policy.pt.")
    policy_group.add_argument("--student_checkpoint", type=str, help="Path to best_student.pt.")
    parser.add_argument("--stats", type=str, default=None, help="Optional path to student_stats.json.")
    parser.add_argument("--xml_path", type=str, required=True, help="Path to the MuJoCo XML/MJCF model.")
    parser.add_argument("--main_camera", type=str, required=True, help="MuJoCo camera name for main_rgb.")
    parser.add_argument("--sub_camera", type=str, default=None, help="MuJoCo camera name for sub_rgb.")
    parser.add_argument("--render_width", type=int, default=256, help="MuJoCo camera render width.")
    parser.add_argument("--render_height", type=int, default=256, help="MuJoCo camera render height.")
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
        "--arm_action_scale",
        type=float,
        default=DEFAULT_ARM_ACTION_SCALE,
        help="Arm action scale used in Isaac Lab JointPositionActionCfg.",
    )
    parser.add_argument(
        "--arm_default_offset",
        type=float,
        nargs=7,
        default=DEFAULT_ARM_OFFSET.tolist(),
        help="Default Franka arm joint positions used as the Isaac Lab action offset.",
    )
    parser.add_argument("--gripper_open_pos", type=float, default=DEFAULT_GRIPPER_OPEN, help="Open finger target.")
    parser.add_argument(
        "--gripper_close_pos", type=float, default=DEFAULT_GRIPPER_CLOSE, help="Closed finger target."
    )
    parser.add_argument(
        "--transport_force_close_gripper",
        action="store_true",
        help="Ignore the policy gripper output while transporting and force the gripper closed.",
    )
    parser.add_argument("--control_rate_hz", type=float, default=20.0, help="Policy control rate.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for checkpoint inference.",
    )
    parser.add_argument(
        "--start_mode",
        choices=("idle", "transport", "release"),
        default="idle",
        help="Initial mode before any key press.",
    )
    return parser.parse_args()


def load_stats(stats_path: str | None, reference_path: str) -> dict:
    if stats_path is None:
        stats_path = str(Path(reference_path).with_name("student_stats.json"))
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def name_to_joint_qpos_addr(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id == -1:
        raise ValueError(f"Joint '{name}' was not found in the MuJoCo model.")
    return int(model.jnt_qposadr[joint_id])


def name_to_actuator_id(model: mujoco.MjModel, name: str) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id == -1:
        raise ValueError(f"Actuator '{name}' was not found in the MuJoCo model.")
    return actuator_id


def render_rgb(renderer: mujoco.Renderer, data: mujoco.MjData, camera_name: str) -> np.ndarray:
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


class PolicyAdapter:
    def __init__(self, stats: dict):
        self.stats = stats
        self.use_sub_camera = bool(stats.get("use_sub_camera", True))

    def infer(self, main_rgb: np.ndarray, qpos: np.ndarray, sub_rgb: np.ndarray | None) -> np.ndarray:
        raise NotImplementedError


class TorchScriptPolicyAdapter(PolicyAdapter):
    def __init__(self, policy_path: str, stats: dict):
        super().__init__(stats)
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()

    def infer(self, main_rgb: np.ndarray, qpos: np.ndarray, sub_rgb: np.ndarray | None) -> np.ndarray:
        main_rgb_t = torch.from_numpy(main_rgb)
        qpos_t = torch.from_numpy(qpos.astype(np.float32))
        with torch.inference_mode():
            if self.use_sub_camera:
                if sub_rgb is None:
                    raise ValueError("This policy expects sub_rgb, but no MuJoCo sub camera was provided.")
                sub_rgb_t = torch.from_numpy(sub_rgb)
                action = self.policy(main_rgb_t, qpos_t, sub_rgb_t)
            else:
                action = self.policy(main_rgb_t, qpos_t)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


class CheckpointPolicyAdapter(PolicyAdapter):
    def __init__(self, checkpoint_path: str, stats: dict, device: str):
        super().__init__(stats)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_args = checkpoint["args"]
        self.image_size = int(stats["image_size"])
        self.qpos_mean = torch.tensor(stats["qpos_mean"], dtype=torch.float32, device=device)
        self.qpos_std = torch.tensor(stats["qpos_std"], dtype=torch.float32, device=device)
        self.device = torch.device(device)
        self.model = ImageQposResNetMLPPolicy(
            qpos_dim=int(stats["qpos_dim"]),
            action_dim=int(stats["action_dim"]),
            image_feature_dim=int(checkpoint_args["image_feature_dim"]),
            mlp_hidden_dims=list(checkpoint_args["mlp_hidden_dims"]),
            dropout=float(checkpoint_args["dropout"]),
            use_sub_camera=self.use_sub_camera,
            freeze_backbone=bool(checkpoint_args.get("freeze_backbone", False)),
            pretrained_backbone=False,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _preprocess_rgb(self, rgb: np.ndarray) -> torch.Tensor:
        rgb_t = torch.from_numpy(rgb).to(self.device).float() / 255.0
        rgb_t = rgb_t.permute(2, 0, 1).unsqueeze(0)
        rgb_t = torch.nn.functional.interpolate(
            rgb_t,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return rgb_t

    def infer(self, main_rgb: np.ndarray, qpos: np.ndarray, sub_rgb: np.ndarray | None) -> np.ndarray:
        main_rgb_t = self._preprocess_rgb(main_rgb)
        qpos_t = torch.from_numpy(qpos.astype(np.float32)).to(self.device).unsqueeze(0)
        qpos_t = (qpos_t - self.qpos_mean) / self.qpos_std
        sub_rgb_t = self._preprocess_rgb(sub_rgb) if (self.use_sub_camera and sub_rgb is not None) else None
        with torch.inference_mode():
            action = self.model(main_rgb_t, qpos_t, sub_rgb_t)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def build_policy_adapter(args, stats: dict) -> PolicyAdapter:
    if args.policy is not None:
        return TorchScriptPolicyAdapter(args.policy, stats)
    return CheckpointPolicyAdapter(args.student_checkpoint, stats, device=args.device)


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


def main():
    args = parse_args()

    if len(args.arm_joint_names) != 7:
        raise ValueError("--arm_joint_names must resolve to exactly 7 arm joints.")
    if len(args.arm_actuator_names) != 7:
        raise ValueError("--arm_actuator_names must resolve to exactly 7 arm actuators.")
    if len(args.arm_default_offset) != 7:
        raise ValueError("--arm_default_offset must provide exactly 7 joint offsets.")

    reference_path = args.policy if args.policy is not None else args.student_checkpoint
    stats = load_stats(args.stats, reference_path)
    policy = build_policy_adapter(args, stats)

    if policy.use_sub_camera and args.sub_camera is None:
        raise ValueError("The selected student policy uses sub_rgb, so --sub_camera must be provided.")

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.render_height, width=args.render_width)

    arm_qpos_addrs = [name_to_joint_qpos_addr(model, name) for name in args.arm_joint_names]
    arm_actuator_ids = [name_to_actuator_id(model, name) for name in args.arm_actuator_names]
    gripper_qpos_addrs = [name_to_joint_qpos_addr(model, name) for name in args.gripper_joint_names]
    gripper_actuator_ids = [name_to_actuator_id(model, name) for name in args.gripper_actuator_names]

    arm_default_offset = np.asarray(args.arm_default_offset, dtype=np.float32)
    last_arm_target = arm_default_offset.copy()
    last_gripper_target = np.full(len(gripper_actuator_ids), args.gripper_open_pos, dtype=np.float32)

    mode_controller = ModeController(args.start_mode)
    control_dt = 1.0 / args.control_rate_hz
    sim_dt = float(model.opt.timestep)
    steps_per_control = max(1, int(round(control_dt / sim_dt)))
    step_counter = 0

    print("[INFO] MuJoCo student runner ready.")
    print("[INFO] Key 1 -> transport policy")
    print("[INFO] Key 2 -> release gripper")
    print("[INFO] Key 0 -> idle / hold current pose")

    with mujoco.viewer.launch_passive(model, data, key_callback=mode_controller.key_callback) as viewer:
        while viewer.is_running():
            if step_counter % steps_per_control == 0:
                qpos = np.asarray(data.qpos[arm_qpos_addrs + gripper_qpos_addrs], dtype=np.float32)

                if mode_controller.mode == "transport":
                    main_rgb = render_rgb(renderer, data, args.main_camera)
                    sub_rgb = render_rgb(renderer, data, args.sub_camera) if policy.use_sub_camera else None
                    policy_action = policy.infer(main_rgb, qpos, sub_rgb)

                    arm_action = np.clip(policy_action[:7], -1.0, 1.0)
                    last_arm_target = arm_default_offset + args.arm_action_scale * arm_action

                    if args.transport_force_close_gripper:
                        last_gripper_target = np.full(
                            len(gripper_actuator_ids), args.gripper_close_pos, dtype=np.float32
                        )
                    else:
                        gripper_open = policy_action[7] >= 0.0
                        gripper_target = args.gripper_open_pos if gripper_open else args.gripper_close_pos
                        last_gripper_target = np.full(len(gripper_actuator_ids), gripper_target, dtype=np.float32)

                elif mode_controller.mode == "release":
                    last_gripper_target = np.full(len(gripper_actuator_ids), args.gripper_open_pos, dtype=np.float32)

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
