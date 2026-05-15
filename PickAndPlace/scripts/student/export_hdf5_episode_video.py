#!/usr/bin/env python3

"""Export one HDF5 demo episode to video files."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Export HDF5 episode images to mp4/gif videos.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the HDF5 dataset.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save exported videos.")
    parser.add_argument("--demo_key", type=str, default=None, help="Episode/demo key. Defaults to the first demo.")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS.")
    parser.add_argument("--step_stride", type=int, default=1, help="Use every N-th frame.")
    parser.add_argument("--max_frames", type=int, default=0, help="Optional maximum number of frames to export.")
    parser.add_argument("--format", type=str, default="mp4", choices=["mp4", "gif"], help="Output video format.")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["main", "sub", "combined", "all"],
        help="Which videos to export.",
    )
    parser.add_argument("--main_rgb_key", type=str, default="main_rgb", help="Main camera dataset key.")
    parser.add_argument("--sub_rgb_key", type=str, default="sub_rgb", help="Sub camera dataset key.")
    return parser.parse_args()


def strip_alpha_if_needed(frames: np.ndarray) -> np.ndarray:
    if frames.shape[-1] == 4:
        return frames[..., :3]
    return frames


def load_frames(group: h5py.Group, key: str, step_stride: int, max_frames: int) -> np.ndarray:
    frames = group[key][::step_stride]
    frames = strip_alpha_if_needed(frames).astype(np.uint8)
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def ensure_same_height(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if left.shape[1] == right.shape[1]:
        return left, right
    target_h = min(left.shape[1], right.shape[1])
    left = left[:, :target_h]
    right = right[:, :target_h]
    return left, right


def write_video(path: Path, frames: np.ndarray, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def main():
    args = parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_path.parent / f"{dataset_path.stem}_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as f:
        demo_keys = sorted(key for key in f.keys() if isinstance(f[key], h5py.Group))
        if not demo_keys:
            raise ValueError(f"No demo groups found in dataset: {dataset_path}")

        demo_key = args.demo_key or demo_keys[0]
        if demo_key not in f:
            raise KeyError(f"Demo key '{demo_key}' not found. Available examples: {demo_keys[:10]}")

        demo = f[demo_key]
        print(f"[INFO] dataset={dataset_path}")
        print(f"[INFO] demo_key={demo_key}")

        main_frames = None
        sub_frames = None

        if args.mode in ("main", "combined", "all"):
            main_frames = load_frames(demo, args.main_rgb_key, args.step_stride, args.max_frames)
            print(f"[INFO] main_rgb shape: {main_frames.shape}")
        if args.mode in ("sub", "combined", "all"):
            sub_frames = load_frames(demo, args.sub_rgb_key, args.step_stride, args.max_frames)
            print(f"[INFO] sub_rgb shape: {sub_frames.shape}")

        suffix = f".{args.format}"
        stem = demo_key

        if args.mode in ("main", "all") and main_frames is not None:
            main_path = output_dir / f"{stem}_main{suffix}"
            write_video(main_path, main_frames, args.fps)
            print(f"[INFO] saved: {main_path}")

        if args.mode in ("sub", "all") and sub_frames is not None:
            sub_path = output_dir / f"{stem}_sub{suffix}"
            write_video(sub_path, sub_frames, args.fps)
            print(f"[INFO] saved: {sub_path}")

        if args.mode in ("combined", "all"):
            if main_frames is None or sub_frames is None:
                raise ValueError("Combined export requires both main and sub camera frames.")
            frame_count = min(len(main_frames), len(sub_frames))
            main_frames = main_frames[:frame_count]
            sub_frames = sub_frames[:frame_count]
            main_frames, sub_frames = ensure_same_height(main_frames, sub_frames)
            combined_frames = np.concatenate([main_frames, sub_frames], axis=2)
            combined_path = output_dir / f"{stem}_combined{suffix}"
            write_video(combined_path, combined_frames, args.fps)
            print(f"[INFO] saved: {combined_path}")


if __name__ == "__main__":
    main()
