#!/usr/bin/env python3

"""Train a vision-based transformer student from recorded teacher trajectories."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer student with HDF5 teacher trajectories.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the recorded HDF5 dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for checkpoints and exports.")
    parser.add_argument("--sequence_length", type=int, default=16, help="Temporal context length.")
    parser.add_argument("--image_size", type=int, default=84, help="Square image size after resizing.")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Optimizer weight decay.")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train split ratio by episodes.")
    parser.add_argument("--num_workers", type=int, default=0, help="PyTorch dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--model_dim", type=int, default=256, help="Transformer hidden size.")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer encoder layers.")
    parser.add_argument("--num_heads", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--camera_feature_dim", type=int, default=128, help="Per-camera feature dimension.")
    parser.add_argument("--proprio_feature_dim", type=int, default=128, help="Proprio encoder feature dimension.")
    parser.add_argument("--max_train_windows", type=int, default=0, help="Optional cap on train windows. 0 disables.")
    parser.add_argument("--max_val_windows", type=int, default=0, help="Optional cap on val windows. 0 disables.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DatasetStats:
    proprio_mean: list[float]
    proprio_std: list[float]
    action_dim: int
    proprio_dim: int
    image_size: int
    sequence_length: int


class Hdf5TrajectoryDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        demo_keys: list[str],
        sequence_length: int,
        image_size: int,
        *,
        proprio_mean: np.ndarray,
        proprio_std: np.ndarray,
        max_windows: int = 0,
    ):
        self.dataset_path = dataset_path
        self.demo_keys = demo_keys
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.proprio_mean = torch.tensor(proprio_mean, dtype=torch.float32)
        self.proprio_std = torch.tensor(proprio_std, dtype=torch.float32)
        self._file: h5py.File | None = None
        self.index: list[tuple[str, int]] = []

        with h5py.File(self.dataset_path, "r") as f:
            for demo_key in self.demo_keys:
                demo = f[demo_key]
                num_steps = int(demo.attrs["num_steps"])
                for step_idx in range(num_steps):
                    self.index.append((demo_key, step_idx))

        if max_windows > 0 and len(self.index) > max_windows:
            self.index = self.index[:max_windows]

    def __len__(self) -> int:
        return len(self.index)

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.dataset_path, "r")
        return self._file

    def __getitem__(self, idx: int):
        demo_key, end_idx = self.index[idx]
        demo = self._get_file()[demo_key]

        start_idx = max(0, end_idx - self.sequence_length + 1)
        pad = self.sequence_length - (end_idx - start_idx + 1)

        main_rgb = torch.from_numpy(demo["main_rgb"][start_idx : end_idx + 1]).float()
        sub_rgb = torch.from_numpy(demo["sub_rgb"][start_idx : end_idx + 1]).float()
        joint_pos = torch.from_numpy(demo["joint_pos"][start_idx : end_idx + 1]).float()
        joint_vel = torch.from_numpy(demo["joint_vel"][start_idx : end_idx + 1]).float()
        last_action = torch.from_numpy(demo["last_action"][start_idx : end_idx + 1]).float()
        target_action = torch.from_numpy(demo["policy_actions"][end_idx]).float()

        proprio = torch.cat([joint_pos, joint_vel, last_action], dim=-1)

        if pad > 0:
            main_rgb = torch.cat([main_rgb[:1].repeat(pad, 1, 1, 1), main_rgb], dim=0)
            sub_rgb = torch.cat([sub_rgb[:1].repeat(pad, 1, 1, 1), sub_rgb], dim=0)
            proprio = torch.cat([proprio[:1].repeat(pad, 1), proprio], dim=0)

        main_rgb = main_rgb[..., :3].permute(0, 3, 1, 2) / 255.0
        sub_rgb = sub_rgb[..., :3].permute(0, 3, 1, 2) / 255.0

        main_rgb = F.interpolate(main_rgb, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        sub_rgb = F.interpolate(sub_rgb, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        proprio = (proprio - self.proprio_mean) / self.proprio_std

        return {
            "main_rgb": main_rgb,
            "sub_rgb": sub_rgb,
            "proprio": proprio,
            "target_action": target_action,
        }


class CameraEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, seq, channels, height, width = images.shape
        x = images.reshape(batch * seq, channels, height, width)
        x = self.backbone(x).flatten(1)
        x = self.proj(x)
        return x.view(batch, seq, -1)


class ProprioEncoder(nn.Module):
    def __init__(self, input_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio)


class TransformerStudent(nn.Module):
    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        sequence_length: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        camera_feature_dim: int,
        proprio_feature_dim: int,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.camera_main = CameraEncoder(camera_feature_dim)
        self.camera_sub = CameraEncoder(camera_feature_dim)
        self.proprio_encoder = ProprioEncoder(proprio_dim, proprio_feature_dim)
        fusion_dim = camera_feature_dim * 2 + proprio_feature_dim
        self.input_proj = nn.Linear(fusion_dim, model_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, sequence_length, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, action_dim),
        )

    def forward(self, main_rgb: torch.Tensor, sub_rgb: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        main_feat = self.camera_main(main_rgb)
        sub_feat = self.camera_sub(sub_rgb)
        proprio_feat = self.proprio_encoder(proprio)
        x = torch.cat([main_feat, sub_feat, proprio_feat], dim=-1)
        x = self.input_proj(x)
        x = x + self.positional_embedding[:, : x.shape[1]]
        x = self.transformer(x)
        return self.head(x[:, -1])


def compute_normalization(dataset_path: str, demo_keys: list[str]) -> tuple[np.ndarray, np.ndarray, int, int]:
    sum_vec = None
    sum_sq_vec = None
    total_steps = 0
    action_dim = 0

    with h5py.File(dataset_path, "r") as f:
        for demo_key in demo_keys:
            demo = f[demo_key]
            joint_pos = demo["joint_pos"][:].astype(np.float32)
            joint_vel = demo["joint_vel"][:].astype(np.float32)
            last_action = demo["last_action"][:].astype(np.float32)
            proprio = np.concatenate([joint_pos, joint_vel, last_action], axis=-1)

            if sum_vec is None:
                sum_vec = proprio.sum(axis=0)
                sum_sq_vec = np.square(proprio).sum(axis=0)
            else:
                sum_vec += proprio.sum(axis=0)
                sum_sq_vec += np.square(proprio).sum(axis=0)
            total_steps += proprio.shape[0]
            action_dim = int(demo["policy_actions"].shape[-1])

    mean = sum_vec / max(total_steps, 1)
    var = sum_sq_vec / max(total_steps, 1) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32), int(mean.shape[0]), action_dim


def split_demos(dataset_path: str, train_split: float, seed: int) -> tuple[list[str], list[str]]:
    with h5py.File(dataset_path, "r") as f:
        demo_keys = sorted(list(f.keys()))
    rng = random.Random(seed)
    rng.shuffle(demo_keys)

    split_idx = max(1, int(len(demo_keys) * train_split))
    split_idx = min(split_idx, len(demo_keys) - 1) if len(demo_keys) > 1 else len(demo_keys)
    train_keys = demo_keys[:split_idx]
    val_keys = demo_keys[split_idx:] if split_idx < len(demo_keys) else demo_keys[:]
    return train_keys, val_keys


def make_dataloader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def run_epoch(model, loader, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        main_rgb = batch["main_rgb"].to(device, non_blocking=True)
        sub_rgb = batch["sub_rgb"].to(device, non_blocking=True)
        proprio = batch["proprio"].to(device, non_blocking=True)
        target_action = batch["target_action"].to(device, non_blocking=True)

        pred_action = model(main_rgb, sub_rgb, proprio)
        loss = F.mse_loss(pred_action, target_action)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = target_action.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val_loss: float, stats: DatasetStats, args):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "stats": asdict(stats),
        "args": vars(args),
    }
    torch.save(payload, path)


class StudentPolicyExport(nn.Module):
    def __init__(
        self,
        model: TransformerStudent,
        proprio_mean: torch.Tensor,
        proprio_std: torch.Tensor,
        image_size: int,
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.register_buffer("proprio_mean", proprio_mean)
        self.register_buffer("proprio_std", proprio_std)

    def forward(self, main_rgb: torch.Tensor, sub_rgb: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        main_rgb = main_rgb.float() / 255.0
        sub_rgb = sub_rgb.float() / 255.0
        if main_rgb.ndim == 4:
            main_rgb = main_rgb.unsqueeze(0)
            sub_rgb = sub_rgb.unsqueeze(0)
            proprio = proprio.unsqueeze(0)
        main_rgb = main_rgb.permute(0, 1, 4, 2, 3)
        sub_rgb = sub_rgb.permute(0, 1, 4, 2, 3)
        main_rgb = F.interpolate(
            main_rgb.reshape(-1, 3, main_rgb.shape[-2], main_rgb.shape[-1]),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        sub_rgb = F.interpolate(
            sub_rgb.reshape(-1, 3, sub_rgb.shape[-2], sub_rgb.shape[-1]),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        main_rgb = main_rgb.view(proprio.shape[0], proprio.shape[1], 3, main_rgb.shape[-2], main_rgb.shape[-1])
        sub_rgb = sub_rgb.view(proprio.shape[0], proprio.shape[1], 3, sub_rgb.shape[-2], sub_rgb.shape[-1])
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        return self.model(main_rgb, sub_rgb, proprio)


def export_policy(model, stats: DatasetStats, output_dir: Path):
    export_wrapper = StudentPolicyExport(
        model,
        torch.tensor(stats.proprio_mean, dtype=torch.float32),
        torch.tensor(stats.proprio_std, dtype=torch.float32),
        stats.image_size,
    ).cpu()
    scripted = torch.jit.script(export_wrapper)
    scripted.save(str(output_dir / "policy.pt"))


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_keys, val_keys = split_demos(args.dataset, args.train_split, args.seed)
    proprio_mean, proprio_std, proprio_dim, action_dim = compute_normalization(args.dataset, train_keys)

    train_dataset = Hdf5TrajectoryDataset(
        args.dataset,
        train_keys,
        args.sequence_length,
        args.image_size,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
        max_windows=args.max_train_windows,
    )
    val_dataset = Hdf5TrajectoryDataset(
        args.dataset,
        val_keys,
        args.sequence_length,
        args.image_size,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
        max_windows=args.max_val_windows,
    )

    print(f"[INFO] Train demos: {len(train_keys)}, train windows: {len(train_dataset)}")
    print(f"[INFO] Val demos: {len(val_keys)}, val windows: {len(val_dataset)}")
    print(f"[INFO] Proprio dim: {proprio_dim}, action dim: {action_dim}")

    train_loader = make_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_dataloader(val_dataset, args.batch_size, args.num_workers, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerStudent(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        sequence_length=args.sequence_length,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        camera_feature_dim=args.camera_feature_dim,
        proprio_feature_dim=args.proprio_feature_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    stats = DatasetStats(
        proprio_mean=proprio_mean.tolist(),
        proprio_std=proprio_std.tolist(),
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        image_size=args.image_size,
        sequence_length=args.sequence_length,
    )
    with open(output_dir / "student_stats.json", "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)
    with open(output_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    best_val_loss = float("inf")
    best_path = output_dir / "best_student.pt"
    epoch_times: list[float] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss = run_epoch(model, val_loader, None, device)
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_minutes, eta_secs = divmod(int(eta_seconds), 60)
        elapsed_minutes, elapsed_secs = divmod(int(sum(epoch_times)), 60)

        print(
            f"[INFO] Epoch {epoch:03d}/{args.epochs:03d} "
            f"| train_loss={train_loss:.6f} "
            f"| val_loss={val_loss:.6f} "
            f"| epoch_time={epoch_time:.1f}s "
            f"| elapsed={elapsed_minutes:02d}m{elapsed_secs:02d}s "
            f"| eta={eta_minutes:02d}m{eta_secs:02d}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, stats, args)
            print(f"[INFO] Saved new best checkpoint to {best_path}")

    checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    export_policy(model, stats, output_dir)
    print(f"[INFO] Exported scripted policy to {output_dir / 'policy.pt'}")


if __name__ == "__main__":
    main()
