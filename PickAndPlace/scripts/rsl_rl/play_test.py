#!/usr/bin/env python3

"""Play an exported ONNX policy in Isaac Lab and dump policy observations/actions to CSV."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", type=str, required=True, help="Path to exported ONNX policy.onnx file.")
parser.add_argument("--trace_csv", type=str, required=True, help="Path to the CSV trace file to write.")
parser.add_argument("--trace_every", type=int, default=1, help="Write one CSV row every N environment steps.")
parser.add_argument("--trace_env_index", type=int, default=0, help="Environment index to trace.")
parser.add_argument("--max_steps", type=int, default=None, help="Optional maximum number of environment steps to record.")
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
        "The 'onnxruntime' package is required for play_test.py. "
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
from policy_trace_utils import CsvTraceWriter


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


def extract_gripper_targets(env, env_index: int) -> np.ndarray:
    gripper_term = env.unwrapped.action_manager.get_term("gripper_action")
    if hasattr(gripper_term, "applied_actions"):
        tensor = gripper_term.applied_actions
    else:
        tensor = gripper_term.processed_actions
    return tensor[env_index].detach().cpu().numpy().astype(np.float32)


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
            "video_folder": os.path.join(log_dir, "videos", "play_test"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if not 0 <= args_cli.trace_env_index < env.unwrapped.num_envs:
        raise ValueError(
            f"--trace_env_index must be in [0, {env.unwrapped.num_envs - 1}], got {args_cli.trace_env_index}."
        )

    providers = choose_onnx_providers(args_cli.device)
    print(f"[INFO]: Loading ONNX policy from: {policy_path}")
    print(f"[INFO]: ONNX Runtime providers: {providers}")
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    trace_writer = CsvTraceWriter(args_cli.trace_csv)
    print(f"[INFO]: Writing Isaac trace CSV to: {trace_writer.path}")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0

    try:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                policy_obs = extract_policy_obs(obs)
                policy_obs_np = policy_obs.detach().cpu().numpy().astype(np.float32)
                actions_np = session.run([output_name], {input_name: policy_obs_np})[0].astype(np.float32)
                actions = torch.from_numpy(actions_np).to(device=env.unwrapped.device, dtype=torch.float32)
                obs, _, _, _ = env.step(actions)

                if timestep % args_cli.trace_every == 0:
                    arm_term = env.unwrapped.action_manager.get_term("arm_action")
                    arm_target = arm_term.processed_actions[args_cli.trace_env_index].detach().cpu().numpy().astype(np.float32)
                    gripper_target = extract_gripper_targets(env, args_cli.trace_env_index)
                    trace_writer.write_row(
                        source="isaac",
                        mode="policy",
                        step=timestep,
                        sim_time=timestep * dt,
                        policy_obs=policy_obs_np[args_cli.trace_env_index],
                        action=actions_np[args_cli.trace_env_index],
                        arm_target=arm_target,
                        gripper_target=gripper_target,
                    )
                    trace_writer.flush()

            timestep += 1

            if args_cli.video and timestep >= args_cli.video_length:
                break
            if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
                break

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        trace_writer.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
