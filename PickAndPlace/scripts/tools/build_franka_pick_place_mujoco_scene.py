#!/usr/bin/env python3

"""Build a MuJoCo scene for the Isaac Lab Franka pick-and-place task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mujoco_scene_utils import DEFAULT_SCENE_XML_PATH, DEFAULT_URDF_PATH, build_pick_place_mujoco_scene


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf_path", type=str, default=str(DEFAULT_URDF_PATH))
    parser.add_argument("--output_xml", type=str, default=str(DEFAULT_SCENE_XML_PATH))
    parser.add_argument("--timestep", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = build_pick_place_mujoco_scene(
        urdf_path=args.urdf_path,
        output_xml_path=args.output_xml,
        timestep=args.timestep,
    )
    print(f"[INFO] Wrote MuJoCo scene: {output_path}")


if __name__ == "__main__":
    main()

