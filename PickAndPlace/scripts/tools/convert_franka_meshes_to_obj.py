#!/usr/bin/env python3

"""Convert bundled Franka description meshes to OBJ files for MuJoCo use."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MESH_ROOT = Path("assets/mujoco/franka_panda/franka_description/meshes")
DEFAULT_PATTERNS = ("collision/*.stl",)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh_root",
        type=str,
        default=str(DEFAULT_MESH_ROOT),
        help="Root directory containing the bundled franka_description meshes.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Directory where OBJ files will be written. Defaults to --mesh_root, "
            "which writes .obj files alongside the source meshes."
        ),
    )
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        default=None,
        help=(
            "Relative glob pattern(s) under --mesh_root to convert. "
            "Can be passed multiple times. Defaults to collision/*.stl."
        ),
    )
    parser.add_argument(
        "--include_visual_dae",
        action="store_true",
        help="Also convert visual/*.dae meshes to OBJ if the current Python environment supports it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing OBJ files.",
    )
    return parser.parse_args()


def _import_trimesh():
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "The 'trimesh' Python package is required to convert meshes. "
            "Install it in the Python environment used to run this script."
        ) from exc
    return trimesh


def _load_as_trimesh(trimesh, src_path: Path):
    loaded = trimesh.load(src_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        meshes = tuple(
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh) and not geometry.is_empty
        )
        if not meshes:
            raise ValueError(f"No mesh geometry found in '{src_path}'.")
        return trimesh.util.concatenate(meshes)
    raise TypeError(f"Unsupported trimesh object '{type(loaded).__name__}' for '{src_path}'.")


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[2] / path).resolve()


def iter_source_paths(mesh_root: Path, patterns: list[str]) -> list[Path]:
    seen: set[Path] = set()
    source_paths: list[Path] = []
    for pattern in patterns:
        for src_path in sorted(mesh_root.glob(pattern)):
            if src_path.is_file() and src_path not in seen:
                seen.add(src_path)
                source_paths.append(src_path)
    return source_paths


def main():
    args = parse_args()
    trimesh = _import_trimesh()

    mesh_root = _resolve_repo_path(args.mesh_root)
    output_root = _resolve_repo_path(args.output_root) if args.output_root else mesh_root

    patterns = list(args.patterns) if args.patterns else list(DEFAULT_PATTERNS)
    if args.include_visual_dae and "visual/*.dae" not in patterns:
        patterns.append("visual/*.dae")

    if not mesh_root.exists():
        raise FileNotFoundError(f"Mesh root does not exist: {mesh_root}")

    source_paths = iter_source_paths(mesh_root, patterns)
    if not source_paths:
        raise FileNotFoundError(f"No meshes matched under '{mesh_root}' for patterns: {patterns}")

    converted_count = 0
    for src_path in source_paths:
        rel_path = src_path.relative_to(mesh_root).with_suffix(".obj")
        dst_path = output_root / rel_path
        if dst_path.exists() and not args.overwrite:
            print(f"[SKIP] Exists: {dst_path}")
            continue

        mesh = _load_as_trimesh(trimesh, src_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(dst_path, file_type="obj")
        converted_count += 1
        print(f"[OK] {src_path} -> {dst_path}")

    print(f"[INFO] Converted {converted_count} mesh file(s) to OBJ.")


if __name__ == "__main__":
    main()
