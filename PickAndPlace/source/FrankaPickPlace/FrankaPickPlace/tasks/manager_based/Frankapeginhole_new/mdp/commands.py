from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class StagedTransportPoseCommand(CommandTerm):
    """Stateful command that visits a fixed waypoint before releasing the final drop target."""

    cfg: "StagedTransportPoseCommandCfg"

    def __init__(self, cfg: "StagedTransportPoseCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        self.current_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.current_command_b[:, 3] = 1.0
        self.drop_command_b = torch.zeros_like(self.current_command_b)
        self.drop_command_b[:, 3] = 1.0
        self.waypoint_command_b = torch.zeros_like(self.current_command_b)
        self.waypoint_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.current_command_b)

        if cfg.waypoint_command_name is None:
            waypoint_pose = torch.tensor(cfg.waypoint_pose, device=self.device)
            self.waypoint_command_b[:, :3] = waypoint_pose[:3]
            waypoint_quat = quat_from_euler_xyz(
                waypoint_pose[3].view(1), waypoint_pose[4].view(1), waypoint_pose[5].view(1)
            )
            self.waypoint_command_b[:, 3:] = quat_unique(waypoint_quat) if cfg.make_quat_unique else waypoint_quat

        self.stage_complete = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.waypoint_hold_elapsed = torch.zeros(self.num_envs, device=self.device)

        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["waypoint_hold_elapsed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stage_complete"] = torch.zeros(self.num_envs, device=self.device)

        self.current_command_b[:] = self.waypoint_command_b

    def __str__(self) -> str:
        msg = "StagedTransportPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tWaypoint pose (base frame): {self.cfg.waypoint_pose}\n"
        msg += f"\tWaypoint hold duration: {self.cfg.hold_duration_s}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.current_command_b

    @property
    def final_command(self) -> torch.Tensor:
        return self.drop_command_b

    def _update_metrics(self):
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.current_command_b[:, :3],
            self.current_command_b[:, 3:],
        )
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)
        self.metrics["waypoint_hold_elapsed"] = self.waypoint_hold_elapsed
        self.metrics["stage_complete"] = self.stage_complete.float()

    def _resample_command(self, env_ids: Sequence[int]):
        self.stage_complete[env_ids] = False
        self.waypoint_hold_elapsed[env_ids] = 0.0
        self._sync_waypoint_command(env_ids)
        self._sync_drop_command(env_ids)
        self.current_command_b[env_ids] = self.waypoint_command_b[env_ids]

    def _update_command(self):
        self._sync_waypoint_command(slice(None))
        self._sync_drop_command(slice(None))

        active_waypoint = ~self.stage_complete
        if torch.any(active_waypoint):
            waypoint_pos_w, _ = combine_frame_transforms(
                self.robot.data.root_pos_w,
                self.robot.data.root_quat_w,
                self.waypoint_command_b[:, :3],
                self.waypoint_command_b[:, 3:],
            )

            object_pos_w = self.object.data.root_pos_w[:, :3]
            xy_distance = torch.norm(object_pos_w[:, :2] - waypoint_pos_w[:, :2], dim=1)
            z_distance = torch.abs(object_pos_w[:, 2] - waypoint_pos_w[:, 2])
            object_speed = torch.norm(self.object.data.root_lin_vel_w, dim=1)

            holding_pose = (
                active_waypoint
                & (object_pos_w[:, 2] > self.cfg.minimal_height)
                & (xy_distance < self.cfg.waypoint_xy_threshold)
                & (z_distance < self.cfg.waypoint_z_threshold)
                & (object_speed < self.cfg.waypoint_max_speed)
            )

            self.waypoint_hold_elapsed[holding_pose] += self._env.step_dt
            self.waypoint_hold_elapsed[active_waypoint & ~holding_pose] = 0.0

            if self.cfg.hold_duration_s <= 0.0:
                finished = holding_pose
            else:
                finished = active_waypoint & (self.waypoint_hold_elapsed >= self.cfg.hold_duration_s)
            self.stage_complete[finished] = True

        self.current_command_b[:] = torch.where(
            self.stage_complete.unsqueeze(-1),
            self.drop_command_b,
            self.waypoint_command_b,
        )

    def _sync_drop_command(self, env_ids: Sequence[int] | slice):
        drop_command = self._env.command_manager.get_command(self.cfg.drop_command_name)
        self.drop_command_b[env_ids] = drop_command[env_ids]

    def _sync_waypoint_command(self, env_ids: Sequence[int] | slice):
        if self.cfg.waypoint_command_name is None:
            return
        waypoint_command = self._env.command_manager.get_command(self.cfg.waypoint_command_name)
        self.waypoint_command_b[env_ids] = waypoint_command[env_ids]

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])


@configclass
class StagedTransportPoseCommandCfg(CommandTermCfg):
    """Configuration for staged transport with a fixed hold waypoint and a final drop target."""

    class_type: type[CommandTerm] = StagedTransportPoseCommand

    asset_name: str = MISSING
    body_name: str = MISSING
    object_name: str = "object"
    waypoint_command_name: str | None = None
    drop_command_name: str = "drop_pose"
    make_quat_unique: bool = False
    waypoint_pose: tuple[float, float, float, float, float, float] | None = None
    hold_duration_s: float = 2.0
    minimal_height: float = 0.04
    waypoint_xy_threshold: float = 0.03
    waypoint_z_threshold: float = 0.03
    waypoint_max_speed: float = 0.05

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/staged_transport_goal"
    )
    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/staged_transport_body"
    )

    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
