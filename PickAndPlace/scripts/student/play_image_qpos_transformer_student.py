#!/usr/bin/env python3

"""Play an image+qpos transformer student policy in Isaac Lab."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
RSL_RL_SCRIPT_DIR = THIS_DIR.parent / "rsl_rl"
if str(RSL_RL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play an image+qpos transformer student policy.")
parser.add_argument("--policy", type=str, required=True, help="Path to the exported student policy.pt file.")
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


installed_version = metadata.version("rsl-rl-lib")


def load_stats(stats_path: str | None, policy_path: str) -> dict:
    if stats_path is None:
        stats_path = str(Path(policy_path).with_name("student_stats.json"))
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_alpha_if_needed(image: torch.Tensor) -> torch.Tensor:
    if image.shape[-1] == 4:
        return image[..., :3]
    return image


def capture_student_inputs(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene["robot"]
    main_camera: Camera = env.scene["main_camera"]
    sub_camera: Camera = env.scene["sub_camera"]

    main_rgb = strip_alpha_if_needed(main_camera.data.output["rgb"]).clone()
    sub_rgb = strip_alpha_if_needed(sub_camera.data.output["rgb"]).clone()
    qpos = robot.data.joint_pos.clone()
    return main_rgb, sub_rgb, qpos


def initialize_history(main_rgb: torch.Tensor, sub_rgb: torch.Tensor, qpos: torch.Tensor, seq_len: int):
    main_hist = main_rgb.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
    sub_hist = sub_rgb.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
    qpos_hist = qpos.unsqueeze(1).repeat(1, seq_len, 1)
    return main_hist, sub_hist, qpos_hist


def update_history(
    main_hist: torch.Tensor,
    sub_hist: torch.Tensor,
    qpos_hist: torch.Tensor,
    main_rgb: torch.Tensor,
    sub_rgb: torch.Tensor,
    qpos: torch.Tensor,
    dones: torch.Tensor,
):
    main_hist = torch.roll(main_hist, shifts=-1, dims=1)
    sub_hist = torch.roll(sub_hist, shifts=-1, dims=1)
    qpos_hist = torch.roll(qpos_hist, shifts=-1, dims=1)

    main_hist[:, -1] = main_rgb
    sub_hist[:, -1] = sub_rgb
    qpos_hist[:, -1] = qpos

    done_mask = dones.to(torch.bool)
    if torch.any(done_mask):
        main_hist[done_mask] = main_rgb[done_mask].unsqueeze(1).repeat(1, main_hist.shape[1], 1, 1, 1)
        sub_hist[done_mask] = sub_rgb[done_mask].unsqueeze(1).repeat(1, sub_hist.shape[1], 1, 1, 1)
        qpos_hist[done_mask] = qpos[done_mask].unsqueeze(1).repeat(1, qpos_hist.shape[1], 1)

    return main_hist, sub_hist, qpos_hist


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.use_cameras = True
    env_cfg.scene.main_camera = make_main_camera_cfg()
    env_cfg.scene.sub_camera = make_sub_camera_cfg()
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    stats = load_stats(args_cli.stats, args_cli.policy)
    sequence_length = int(stats["sequence_length"])

    policy_path = os.path.abspath(args_cli.policy)
    print(f"[INFO] Loading student policy from: {policy_path}")
    print(f"[INFO] Using stats: sequence_length={sequence_length}, image_size={stats['image_size']}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.path.dirname(policy_path), "videos", "student_play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during student playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    student_policy = torch.jit.load(policy_path, map_location=env.unwrapped.device)
    student_policy.eval()

    if env.unwrapped.sim.has_rtx_sensors():
        print("[INFO] Warming up camera sensors...")
        for _ in range(3):
            env.unwrapped.sim.render()

    _ = env.get_observations()
    main_rgb, sub_rgb, qpos = capture_student_inputs(env.unwrapped)
    main_hist, sub_hist, qpos_hist = initialize_history(main_rgb, sub_rgb, qpos, sequence_length)

    timestep = 0
    dt = env.unwrapped.step_dt

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            actions = student_policy(main_hist, sub_hist, qpos_hist)
            _, _, dones, _ = env.step(actions)

        main_rgb, sub_rgb, qpos = capture_student_inputs(env.unwrapped)
        main_hist, sub_hist, qpos_hist = update_history(main_hist, sub_hist, qpos_hist, main_rgb, sub_rgb, qpos, dones)

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
