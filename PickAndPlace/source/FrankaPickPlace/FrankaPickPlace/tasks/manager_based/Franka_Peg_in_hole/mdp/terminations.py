# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from .rewards import _hole_target_world, _peg_tip_world, _peg_upright_projection

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def peg_inserted(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    success_depth: float,
    command_name: str,
    peg_half_length: float,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Terminate an episode once the peg is inserted deeply and upright."""
    peg_tip_w = _peg_tip_world(env, peg_half_length=peg_half_length, peg_cfg=peg_cfg)
    hole_target_w = _hole_target_world(env, command_name=command_name)
    upright = _peg_upright_projection(env, peg_cfg=peg_cfg)

    xy_distance = torch.norm(peg_tip_w[:, :2] - hole_target_w[:, :2], dim=1)
    depth = hole_target_w[:, 2] - peg_tip_w[:, 2]
    return (xy_distance < xy_threshold) & (depth > success_depth) & (upright > 0.95)
