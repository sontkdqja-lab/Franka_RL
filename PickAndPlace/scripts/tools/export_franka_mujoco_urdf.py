#!/usr/bin/env python3

"""Export a MuJoCo-usable local Franka URDF package from Isaac Sim assets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_URDF_CANDIDATES = [
    Path("/home/beombu/Library/IsaacLab/source/isaaclab/isaaclab/controllers/config/data/lula_franka_gen.urdf"),
]

DEFAULT_MESH_ROOT_CANDIDATES = [
    Path(
        "/home/beombu/anaconda3/envs/franka/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description"
    ),
    Path(
        "/home/beombu/anaconda3/envs/rl311/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description"
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets/mujoco/franka_panda",
        help="Directory where the local MuJoCo-usable Franka package will be written.",
    )
    return parser.parse_args()


def find_existing_path(candidates: list[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {description}. Checked: {candidates}")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (repo_root / args.output_dir).resolve()

    src_urdf = find_existing_path(DEFAULT_URDF_CANDIDATES, "Franka URDF")
    src_mesh_root = find_existing_path(DEFAULT_MESH_ROOT_CANDIDATES, "franka_description mesh package")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dst_mesh_root = output_dir / "franka_description"
    shutil.copytree(src_mesh_root, dst_mesh_root)

    urdf_text = src_urdf.read_text(encoding="utf-8")
    urdf_text = urdf_text.replace("package://franka_description/", "franka_description/")
    for mesh_name in ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "finger"]:
        urdf_text = urdf_text.replace(
            f"franka_description/meshes/visual/{mesh_name}.dae",
            f"franka_description/meshes/collision/{mesh_name}.stl",
        )
    dst_urdf = output_dir / "lula_franka_mujoco.urdf"
    dst_urdf.write_text(urdf_text, encoding="utf-8")

    print(f"[INFO] Wrote URDF: {dst_urdf}")
    print(f"[INFO] Wrote meshes: {dst_mesh_root}")


if __name__ == "__main__":
    main()
