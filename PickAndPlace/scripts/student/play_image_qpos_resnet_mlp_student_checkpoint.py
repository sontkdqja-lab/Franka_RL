#!/usr/bin/env python3

"""Play an image+qpos ResNet-MLP student policy from best_student.pt or policy.pt."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
RSL_RL_SCRIPT_DIR = THIS_DIR.parent / "rsl_rl"
if str(RSL_RL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play an image+qpos ResNet-MLP student checkpoint or TorchScript policy.")
policy_group = parser.add_mutually_exclusive_group(required=True)
policy_group.add_argument("--student_checkpoint", type=str, help="Path to best_student.pt checkpoint.")
policy_group.add_argument("--policy", type=str, help="Path to exported student policy.pt file.")
parser.add_argument("--stats", type=str, default=None, help="Optional path to student_stats.json.")
parser.add_argument("--video", action="store_true", default=False, help="Record video during playback.")
parser.add_argument("--video_length", type=int, default=400, help="Number of steps to record when video is enabled.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib.metadata as metadata

from isaaclab.assets import Articulation
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.sensors import Camera
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import FrankaPickPlace.tasks  # noqa: F401
from FrankaPickPlace.tasks.manager_based.frankapickplace.frankapickplace_env_cfg import (
    make_main_camera_cfg,
    make_sub_camera_cfg,
)
from train_image_qpos_resnet_mlp_student import ImageQposResNetMLPPolicy


installed_version = metadata.version("rsl-rl-lib")


def load_stats(stats_path: str | None, source_path: str) -> dict:
    if stats_path is None:
        stats_path = str(Path(source_path).with_name("student_stats.json"))
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_alpha_if_needed(image: torch.Tensor) -> torch.Tensor:
    if image.shape[-1] == 4:
        return image[..., :3]
    return image


def preprocess_rgb(rgb: torch.Tensor, image_size: int) -> torch.Tensor:
    rgb = rgb.float() / 255.0
    rgb = rgb.permute(0, 3, 1, 2)
    rgb = F.interpolate(rgb, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return rgb


def capture_student_inputs(env, use_sub_camera: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    robot: Articulation = env.scene["robot"]
    main_camera: Camera = env.scene["main_camera"]

    main_rgb = strip_alpha_if_needed(main_camera.data.output["rgb"]).clone()
    qpos = robot.data.joint_pos.clone()

    if use_sub_camera:
        sub_camera: Camera = env.scene["sub_camera"]
        sub_rgb = strip_alpha_if_needed(sub_camera.data.output["rgb"]).clone()
    else:
        sub_rgb = None

    return main_rgb, qpos, sub_rgb


def build_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[ImageQposResNetMLPPolicy, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint["args"]
    stats = checkpoint["stats"]

    use_sub_camera = not checkpoint_args.get("disable_sub_camera", False)
    model = ImageQposResNetMLPPolicy(
        qpos_dim=int(stats["qpos_dim"]),
        action_dim=int(stats["action_dim"]),
        image_feature_dim=int(checkpoint_args["image_feature_dim"]),
        mlp_hidden_dims=list(checkpoint_args["mlp_hidden_dims"]),
        dropout=float(checkpoint_args["dropout"]),
        use_sub_camera=use_sub_camera,
        freeze_backbone=bool(checkpoint_args.get("freeze_backbone", False)),
        pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.use_cameras = True
    env_cfg.scene.main_camera = make_main_camera_cfg()

    source_path = os.path.abspath(args_cli.policy or args_cli.student_checkpoint)
    stats = load_stats(args_cli.stats, source_path)
    use_sub_camera = bool(stats.get("use_sub_camera", True))
    image_size = int(stats["image_size"])
    if use_sub_camera:
        env_cfg.scene.sub_camera = make_sub_camera_cfg()
    else:
        env_cfg.scene.sub_camera = None

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.policy is not None:
        print(f"[INFO] Loading student TorchScript policy from: {source_path}")
    else:
        print(f"[INFO] Loading student checkpoint from: {source_path}")
    print(f"[INFO] Using stats: image_size={stats['image_size']}, use_sub_camera={use_sub_camera}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.path.dirname(source_path), "videos", "student_play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during student playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = env.unwrapped.device
    if args_cli.policy is not None:
        student_policy = torch.jit.load(source_path, map_location=device)
        student_policy.eval()
        student_model = None
        qpos_mean = None
        qpos_std = None
    else:
        qpos_mean = torch.tensor(stats["qpos_mean"], dtype=torch.float32, device=device)
        qpos_std = torch.tensor(stats["qpos_std"], dtype=torch.float32, device=device)
        student_model, _ = build_model_from_checkpoint(source_path, device)
        student_policy = None

    if env.unwrapped.sim.has_rtx_sensors():
        print("[INFO] Warming up camera sensors...")
        for _ in range(3):
            env.unwrapped.sim.render()

    _ = env.get_observations()
    dt = env.unwrapped.step_dt
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()
        main_rgb, qpos, sub_rgb = capture_student_inputs(env.unwrapped, use_sub_camera)

        with torch.inference_mode():
            if student_policy is not None:
                if use_sub_camera:
                    actions = student_policy(main_rgb, qpos, sub_rgb)
                else:
                    actions = student_policy(main_rgb, qpos)
            else:
                main_rgb = preprocess_rgb(main_rgb, image_size)
                qpos = (qpos.float() - qpos_mean) / qpos_std
                if sub_rgb is not None:
                    sub_rgb = preprocess_rgb(sub_rgb, image_size)
                actions = student_model(main_rgb, qpos, sub_rgb)
            _, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
