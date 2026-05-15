#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


POLICY_OBS_LABELS = (
    [f"obs_joint_pos_rel_{idx}" for idx in range(9)]
    + [f"obs_joint_vel_{idx}" for idx in range(9)]
    + [f"obs_object_pos_b_{idx}" for idx in range(3)]
    + [f"obs_place_target_{idx}" for idx in range(7)]
    + ["obs_transport_phase_0"]
    + [f"obs_last_action_{idx}" for idx in range(8)]
)
ACTION_LABELS = [f"action_{idx}" for idx in range(8)]
ARM_TARGET_LABELS = [f"arm_target_{idx}" for idx in range(7)]
GRIPPER_TARGET_LABELS = [f"gripper_target_{idx}" for idx in range(2)]
FIELDNAMES = ["source", "mode", "step", "sim_time"] + POLICY_OBS_LABELS + ACTION_LABELS + ARM_TARGET_LABELS + GRIPPER_TARGET_LABELS


def _flatten(values: Iterable[float] | np.ndarray, expected_len: int, label: str) -> list[float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size != expected_len:
        raise ValueError(f"{label} expected {expected_len} values, got {array.size}.")
    return [float(value) for value in array]


class CsvTraceWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()

    def write_row(
        self,
        source: str,
        mode: str,
        step: int,
        sim_time: float,
        policy_obs: Iterable[float] | np.ndarray,
        action: Iterable[float] | np.ndarray,
        arm_target: Iterable[float] | np.ndarray,
        gripper_target: Iterable[float] | np.ndarray,
    ) -> None:
        row = {
            "source": source,
            "mode": mode,
            "step": int(step),
            "sim_time": float(sim_time),
        }
        for key, value in zip(POLICY_OBS_LABELS, _flatten(policy_obs, len(POLICY_OBS_LABELS), "policy_obs"), strict=True):
            row[key] = value
        for key, value in zip(ACTION_LABELS, _flatten(action, len(ACTION_LABELS), "action"), strict=True):
            row[key] = value
        for key, value in zip(ARM_TARGET_LABELS, _flatten(arm_target, len(ARM_TARGET_LABELS), "arm_target"), strict=True):
            row[key] = value
        for key, value in zip(
            GRIPPER_TARGET_LABELS, _flatten(gripper_target, len(GRIPPER_TARGET_LABELS), "gripper_target"), strict=True
        ):
            row[key] = value
        self._writer.writerow(row)

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()
