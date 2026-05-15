#!/usr/bin/env python3

"""Plot Isaac/MuJoCo policy trace CSV files into grouped PNG figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUPS = {
    "obs_joint_pos_rel": ("obs_joint_pos_rel_",),
    "obs_joint_vel": ("obs_joint_vel_",),
    "obs_object_pos_b": ("obs_object_pos_b_",),
    "obs_place_target": ("obs_place_target_",),
    "obs_transport_phase": ("obs_transport_phase_",),
    "obs_last_action": ("obs_last_action_",),
    "action": ("action_",),
    "arm_target": ("arm_target_",),
    "gripper_target": ("gripper_target_",),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, required=True, help="Path to the CSV trace file.")
    parser.add_argument("--outdir", type=str, default=None, help="Directory to save PNG figures into.")
    parser.add_argument("--mode", type=str, default=None, help="Optional mode filter, for example 'policy'.")
    parser.add_argument("--max_legend", type=int, default=12, help="Maximum number of legend entries to show per plot.")
    return parser.parse_args()


def load_rows(csv_path: Path, mode_filter: str | None) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if mode_filter is None:
        return rows
    return [row for row in rows if row.get("mode") == mode_filter]


def collect_columns(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    if not rows:
        return {}
    columns: dict[str, list[float]] = {}
    for key in rows[0]:
        if key in {"source", "mode"}:
            continue
        columns[key] = [float(row[key]) for row in rows]
    return {key: np.asarray(values, dtype=np.float32) for key, values in columns.items()}


def matching_columns(columns: dict[str, np.ndarray], prefixes: tuple[str, ...]) -> list[str]:
    matches = [name for name in columns if any(name.startswith(prefix) for prefix in prefixes)]
    return sorted(matches)


def main():
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir is not None else csv_path.with_suffix("")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path, args.mode)
    if not rows:
        raise ValueError("No rows found to plot. Check the CSV path and optional --mode filter.")

    columns = collect_columns(rows)
    time_values = columns.get("sim_time")
    if time_values is None:
        raise ValueError("CSV is missing the 'sim_time' column.")

    for group_name, prefixes in GROUPS.items():
        group_columns = matching_columns(columns, prefixes)
        if not group_columns:
            continue

        figure, axis = plt.subplots(figsize=(12, 5))
        for column_name in group_columns:
            axis.plot(time_values, columns[column_name], label=column_name, linewidth=1.2)
        axis.set_title(group_name)
        axis.set_xlabel("sim_time [s]")
        axis.set_ylabel("value")
        axis.grid(True, alpha=0.3)
        if len(group_columns) <= args.max_legend:
            axis.legend(loc="best", fontsize=8)
        figure.tight_layout()
        figure.savefig(outdir / f"{group_name}.png", dpi=150)
        plt.close(figure)

    print(f"[INFO] Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
