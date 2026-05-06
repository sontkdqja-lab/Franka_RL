#!/usr/bin/env python3

"""Record successful teacher trajectories for vision-based student training."""

import argparse
import importlib.metadata as metadata
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Record successful teacher trajectories with camera observations.")
parser.add_argument("--num_demos", type=int, default=100, help="Number of successful demonstrations to save.")
parser.add_argument("--output", type=str, required=True, help="Output HDF5 file path.")
parser.add_argument("--save_failed", action="store_true", help="Also save failed or timed-out episodes.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# camera extraction requires cameras to be enabled in the app
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.sensors import Camera
from isaaclab.utils.assets import retrieve_file_path

try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import FrankaPickPlace.tasks  # noqa: F401
from FrankaPickPlace.tasks.manager_based.frankapickplace.frankapickplace_env_cfg import (
    make_main_camera_cfg,
    make_sub_camera_cfg,
)


installed_version = metadata.version("rsl-rl-lib")


def to_numpy(tensor: torch.Tensor, *, dtype: np.dtype | None = None) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if dtype is not None:
        array = array.astype(dtype)
    return array


def strip_alpha_if_needed(image: torch.Tensor) -> torch.Tensor:
    if image.shape[-1] == 4:
        return image[..., :3]
    return image


@dataclass
class EpisodeSample:
    data: dict[str, list[np.ndarray | float | int]]

    @classmethod
    def create(cls) -> "EpisodeSample":
        return cls(defaultdict(list))


class Hdf5EpisodeWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(output_path, "w")
        self._demo_index = 0
        self.success_count = 0
        self.total_count = 0

    def close(self):
        self._file.attrs["num_demos"] = self._demo_index
        self._file.attrs["num_success"] = self.success_count
        self._file.close()

    def write_episode(self, episode: EpisodeSample, *, success: bool, time_out: bool, total_reward: float):
        group = self._file.create_group(f"demo_{self._demo_index:05d}")
        self._demo_index += 1
        self.total_count += 1
        if success:
            self.success_count += 1

        for key, values in episode.data.items():
            if not values:
                continue
            first = values[0]
            if isinstance(first, np.ndarray):
                group.create_dataset(key, data=np.stack(values, axis=0), compression="gzip")
            else:
                group.create_dataset(key, data=np.asarray(values))

        group.attrs["num_steps"] = len(episode.data.get("rewards", []))
        group.attrs["success"] = bool(success)
        group.attrs["time_out"] = bool(time_out)
        group.attrs["total_reward"] = float(total_reward)


def resolve_output_path(output_arg: str) -> str:
    path = Path(output_arg).expanduser()
    if path.exists() and path.is_dir():
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return str(path / f"{timestamp}.hdf5")
    if path.suffix.lower() == ".hdf5":
        return str(path)
    if not path.suffix:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return str(path / f"{timestamp}.hdf5")
    return str(path)


def capture_auto_open_mask(env) -> torch.Tensor:
    try:
        gripper_term = env.action_manager.get_term("gripper_action")
        if hasattr(gripper_term, "_compute_auto_open_mask"):
            return gripper_term._compute_auto_open_mask().clone()
    except Exception:
        pass
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def capture_step_inputs(env, obs, actions: torch.Tensor) -> dict[str, np.ndarray]:
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object"]
    main_camera: Camera = env.scene["main_camera"]
    sub_camera: Camera = env.scene["sub_camera"]

    main_rgb = strip_alpha_if_needed(main_camera.data.output["rgb"])
    sub_rgb = strip_alpha_if_needed(sub_camera.data.output["rgb"])
    transport_target = env.command_manager.get_command("transport_target")
    box_pose = env.command_manager.get_command("box_pose")
    stage_complete = env.command_manager.get_term("transport_target").stage_complete.float().unsqueeze(1)
    auto_open_mask = capture_auto_open_mask(env).float().unsqueeze(1)

    return {
        "policy_obs": to_numpy(obs["policy"], dtype=np.float32),
        "joint_pos": to_numpy(robot.data.joint_pos, dtype=np.float32),
        "joint_vel": to_numpy(robot.data.joint_vel, dtype=np.float32),
        "last_action": to_numpy(env.action_manager.action, dtype=np.float32),
        "policy_actions": to_numpy(actions, dtype=np.float32),
        "main_rgb": to_numpy(main_rgb),
        "sub_rgb": to_numpy(sub_rgb),
        "object_pos_w": to_numpy(obj.data.root_pos_w[:, :3], dtype=np.float32),
        "object_quat_w": to_numpy(obj.data.root_quat_w, dtype=np.float32),
        "transport_target": to_numpy(transport_target, dtype=np.float32),
        "box_pose": to_numpy(box_pose, dtype=np.float32),
        "stage_complete": to_numpy(stage_complete, dtype=np.float32),
        "auto_open_mask": to_numpy(auto_open_mask, dtype=np.float32),
    }


def append_step(
    episode: EpisodeSample,
    step_inputs: dict[str, np.ndarray],
    env_idx: int,
    reward: float,
    done: bool,
    terminated: bool,
    time_out: bool,
):
    for key, value in step_inputs.items():
        episode.data[key].append(value[env_idx].copy())
    episode.data["rewards"].append(np.float32(reward))
    episode.data["dones"].append(np.uint8(done))
    episode.data["terminated"].append(np.uint8(terminated))
    episode.data["time_outs"].append(np.uint8(time_out))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg.use_cameras = True
    env_cfg.scene.main_camera = make_main_camera_cfg()
    env_cfg.scene.sub_camera = make_sub_camera_cfg()
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            raise FileNotFoundError("No published checkpoint is available for this task.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    output_path = resolve_output_path(args_cli.output)

    print(f"[INFO] Recording teacher trajectories from checkpoint: {resume_path}")
    print(f"[INFO] Output dataset: {output_path}")
    print(f"[INFO] Requested successful demos: {args_cli.num_demos}")
    print(f"[INFO] Num envs: {env_cfg.scene.num_envs}")

    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if env.unwrapped.sim.has_rtx_sensors():
        print("[INFO] Warming up camera sensors...")
        for _ in range(3):
            env.unwrapped.sim.render()

    obs = env.get_observations()
    episode_buffers = [EpisodeSample.create() for _ in range(env.num_envs)]
    episode_rewards = np.zeros(env.num_envs, dtype=np.float64)
    writer = Hdf5EpisodeWriter(output_path)
    last_reported_successes = 0
    total_steps = 0

    try:
        while simulation_app.is_running() and writer.success_count < args_cli.num_demos:
            with torch.inference_mode():
                actions = policy(obs)
                step_inputs = capture_step_inputs(env.unwrapped, obs, actions)
                obs, rewards, dones, _ = env.step(actions)
                policy.reset(dones)

            terminated = env.unwrapped.reset_terminated.clone()
            time_outs = env.unwrapped.reset_time_outs.clone()
            total_steps += 1

            reward_np = to_numpy(rewards, dtype=np.float32)
            done_np = to_numpy(dones.to(torch.bool))
            terminated_np = to_numpy(terminated)
            time_out_np = to_numpy(time_outs)

            for env_idx in range(env.num_envs):
                append_step(
                    episode_buffers[env_idx],
                    step_inputs,
                    env_idx,
                    float(reward_np[env_idx]),
                    bool(done_np[env_idx]),
                    bool(terminated_np[env_idx]),
                    bool(time_out_np[env_idx]),
                )
                episode_rewards[env_idx] += float(reward_np[env_idx])

                if done_np[env_idx]:
                    success = bool(terminated_np[env_idx] and not time_out_np[env_idx])
                    if success or args_cli.save_failed:
                        writer.write_episode(
                            episode_buffers[env_idx],
                            success=success,
                            time_out=bool(time_out_np[env_idx]),
                            total_reward=float(episode_rewards[env_idx]),
                        )
                        print(
                            f"[INFO] Stored episode {writer.total_count}: "
                            f"success={success}, time_out={bool(time_out_np[env_idx])}, "
                            f"steps={len(episode_buffers[env_idx].data.get('rewards', []))}, "
                            f"return={episode_rewards[env_idx]:.2f}"
                        )
                    episode_buffers[env_idx] = EpisodeSample.create()
                    episode_rewards[env_idx] = 0.0

            if writer.success_count and writer.success_count % 10 == 0 and writer.success_count != last_reported_successes:
                print(
                    f"[INFO] Saved {writer.success_count} successful demos "
                    f"({writer.total_count} total stored) to {output_path}"
                )
                last_reported_successes = writer.success_count
            elif total_steps % 200 == 0:
                print(
                    f"[INFO] Running... env_steps={total_steps}, "
                    f"successes={writer.success_count}, stored={writer.total_count}"
                )

    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
