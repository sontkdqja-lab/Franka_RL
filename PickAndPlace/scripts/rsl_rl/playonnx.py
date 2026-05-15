#!/usr/bin/env python3

"""Play an exported ONNX policy directly in Isaac Lab."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", type=str, required=True, help="Path to exported ONNX policy.onnx file.")
parser.add_argument("--video", action="store_true", default=False, help="Record video during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--fixed_cube_xy",
    type=float,
    nargs=2,
    default=None,
    metavar=("X", "Y"),
    help="Optional fixed cube XY position for play. Overrides the task reset grid for the object.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib.metadata as metadata
import os
import time

import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError as exc:
    raise RuntimeError(
        "The 'onnxruntime' package is required for playonnx.py. "
        "Install 'onnxruntime' or 'onnxruntime-gpu' in the Python environment used to run this script."
    ) from exc

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import FrankaPickPlace.tasks  # noqa: F401


installed_version = metadata.version("rsl-rl-lib")


def extract_policy_obs(obs):
    if hasattr(obs, "keys") and "policy" in obs.keys():
        return obs["policy"]
    if isinstance(obs, dict) and "policy" in obs:
        return obs["policy"]
    return obs


def choose_onnx_providers(device: str | None) -> list[str]:
    available = set(ort.get_available_providers())
    if device is not None and device.startswith("cuda") and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.fixed_cube_xy is not None and hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_object_position"):
        fixed_x, fixed_y = (float(args_cli.fixed_cube_xy[0]), float(args_cli.fixed_cube_xy[1]))
        env_cfg.events.reset_object_position.params["grid_x"] = [fixed_x]
        env_cfg.events.reset_object_position.params["grid_y"] = [fixed_y]
        env_cfg.events.reset_object_position.params["pose_range"] = {"z": (0.0, 0.0)}
        env_cfg.events.reset_object_position.params["velocity_range"] = {}
        print(f"[INFO] Fixed cube XY for play: x={fixed_x}, y={fixed_y}")

    policy_path = retrieve_file_path(args_cli.policy)
    policy_path = os.path.abspath(policy_path)
    log_dir = os.path.dirname(policy_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_onnx"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    providers = choose_onnx_providers(args_cli.device)
    print(f"[INFO]: Loading ONNX policy from: {policy_path}")
    print(f"[INFO]: ONNX Runtime providers: {providers}")
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            policy_obs = extract_policy_obs(obs)
            policy_obs_np = policy_obs.detach().cpu().numpy().astype(np.float32)
            actions_np = session.run([output_name], {input_name: policy_obs_np})[0]
            actions = torch.from_numpy(actions_np).to(device=env.unwrapped.device, dtype=torch.float32)
            obs, _, dones, _ = env.step(actions)

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
