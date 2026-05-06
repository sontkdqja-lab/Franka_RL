from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs.mdp.actions import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers import ActionTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _stage_complete_mask(env: ManagerBasedRLEnv, command_name: str | None) -> torch.Tensor:
    """Return the staged-command completion mask if the command term exposes one."""
    if command_name is None:
        return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    term = env.command_manager.get_term(command_name)
    if hasattr(term, "stage_complete"):
        return term.stage_complete
    return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)


class AutoOpenBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary gripper action that auto-opens near the box target."""

    cfg: "AutoOpenBinaryJointPositionActionCfg"

    def __init__(self, cfg: "AutoOpenBinaryJointPositionActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._release_latched = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._applied_actions = torch.zeros_like(self._processed_actions)

    @property
    def applied_actions(self) -> torch.Tensor:
        """The actual joint targets sent to the simulator after the auto-open override."""
        return self._applied_actions

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._release_latched[:] = False
            self._applied_actions.zero_()
        else:
            self._release_latched[env_ids] = False
            self._applied_actions[env_ids] = 0.0

    def apply_actions(self):
        commands = self._processed_actions.clone()
        release_mask = self._compute_auto_open_mask()

        if self.cfg.latch_open:
            self._release_latched |= release_mask
            release_mask = self._release_latched

        if torch.any(release_mask):
            commands[release_mask] = self._open_command

        self._applied_actions[:] = commands
        self._asset.set_joint_position_target(commands, joint_ids=self._joint_ids)

    def _compute_auto_open_mask(self) -> torch.Tensor:
        env = self._env
        robot: RigidObject = env.scene[self.cfg.asset_name]
        object: RigidObject = env.scene[self.cfg.object_cfg.name]
        command = env.command_manager.get_command(self.cfg.command_name)

        target_pos_b = command[:, :3]
        target_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
        )

        object_pos_w = object.data.root_pos_w[:, :3]
        xy_distance = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
        height_distance = torch.abs(object_pos_w[:, 2] - target_pos_w[:, 2])
        near_box = (xy_distance < self.cfg.target_xy_threshold) & (height_distance < self.cfg.target_height_threshold)

        return near_box & _stage_complete_mask(env, self.cfg.staged_command_name)


@configclass
class AutoOpenBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Configuration for a binary gripper action that auto-opens near the final target."""

    class_type: type[ActionTerm] = AutoOpenBinaryJointPositionAction

    command_name: str = "box_pose"
    staged_command_name: str | None = "transport_target"
    target_xy_threshold: float = 0.01
    target_height_threshold: float = 0.02
    latch_open: bool = True
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
