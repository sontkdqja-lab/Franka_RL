#!/usr/bin/env python3

"""Convert ACT dataset_stats.pkl to a JSON file for cross-environment playback."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ACT dataset_stats.pkl to dataset_stats.json.")
    parser.add_argument("--input", type=str, required=True, help="Path to dataset_stats.pkl.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path. Defaults to the input path with a .json suffix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if args.output is None:
        output_path = input_path.with_suffix(".json")
    else:
        output_path = Path(args.output).expanduser().resolve()

    with open(input_path, "rb") as f:
        stats = pickle.load(f)

    jsonable_stats = to_jsonable(stats)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(jsonable_stats, f, indent=2)

    print(f"[INFO] Wrote JSON stats to: {output_path}")


if __name__ == "__main__":
    main()
