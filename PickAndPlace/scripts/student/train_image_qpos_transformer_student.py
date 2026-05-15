#!/usr/bin/env python3

"""Train a privileged-teacher / image+qpos student transformer from HDF5 trajectories."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a transformer student from teacher trajectories using images and qpos only."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 teacher dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for checkpoints and exports.")
    parser.add_argument("--sequence_length", type=int, default=16, help="Temporal context length.")
    parser.add_argument("--image_size", type=int, default=84, help="Square image size after resizing.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--train_split", type=float, default=0.8, help="Episode split ratio.")
    parser.add_argument("--num_workers", type=int, default=0, help="PyTorch dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--model_dim", type=int, default=192, help="Transformer hidden size.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer blocks.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="Feedforward hidden width ratio.")
    parser.add_argument("--camera_feature_dim", type=int, default=96, help="Feature size per camera stream.")
    parser.add_argument("--qpos_feature_dim", type=int, default=64, help="Encoded qpos feature size.")
    parser.add_argument("--max_train_windows", type=int, default=0, help="Optional cap on train windows.")
    parser.add_argument("--max_val_windows", type=int, default=0, help="Optional cap on val windows.")
    parser.add_argument("--main_rgb_key", type=str, default="main_rgb", help="Main camera dataset key.")
    parser.add_argument("--sub_rgb_key", type=str, default="sub_rgb", help="Sub camera dataset key.")
    parser.add_argument("--qpos_key", type=str, default="joint_pos", help="Joint position dataset key.")
    parser.add_argument("--action_key", type=str, default="policy_actions", help="Teacher action dataset key.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_output_dir(output_root: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


@dataclass
class DatasetStats:
    qpos_mean: list[float]
    qpos_std: list[float]
    qpos_dim: int
    action_dim: int
    image_size: int
    sequence_length: int


def list_demo_keys(dataset_path: str, required_keys: list[str]) -> list[str]:
    with h5py.File(dataset_path, "r") as f:
        demo_keys = []
        for key in sorted(f.keys()):
            group = f[key]
            if isinstance(group, h5py.Group) and all(req in group for req in required_keys):
                demo_keys.append(key)
    if not demo_keys:
        raise ValueError(f"No demo groups with required keys {required_keys} were found in dataset: {dataset_path}")
    return demo_keys


def split_demo_keys(demo_keys: list[str], train_split: float, seed: int) -> tuple[list[str], list[str]]:
    shuffled = demo_keys[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, shuffled
    split_idx = max(1, int(len(shuffled) * train_split))
    split_idx = min(split_idx, len(shuffled) - 1)
    return shuffled[:split_idx], shuffled[split_idx:]


def compute_qpos_normalization(
    dataset_path: str, demo_keys: list[str], qpos_key: str, action_key: str
) -> tuple[np.ndarray, np.ndarray, int, int]:
    qpos_sum = None
    qpos_sq_sum = None
    total_steps = 0
    action_dim = 0

    with h5py.File(dataset_path, "r") as f:
        for demo_key in demo_keys:
            demo = f[demo_key]
            qpos = demo[qpos_key][:].astype(np.float32)
            actions = demo[action_key][:]
            if qpos_sum is None:
                qpos_sum = qpos.sum(axis=0)
                qpos_sq_sum = np.square(qpos).sum(axis=0)
            else:
                qpos_sum += qpos.sum(axis=0)
                qpos_sq_sum += np.square(qpos).sum(axis=0)
            total_steps += qpos.shape[0]
            action_dim = int(actions.shape[-1])

    mean = qpos_sum / max(total_steps, 1)
    var = qpos_sq_sum / max(total_steps, 1) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1.0e-6))
    return mean.astype(np.float32), std.astype(np.float32), int(mean.shape[0]), action_dim


def strip_alpha_if_needed(images: np.ndarray) -> np.ndarray:
    if images.shape[-1] == 4:
        return images[..., :3]
    return images


class ImageQposDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        demo_keys: list[str],
        sequence_length: int,
        image_size: int,
        main_rgb_key: str,
        sub_rgb_key: str,
        qpos_key: str,
        action_key: str,
        qpos_mean: np.ndarray,
        qpos_std: np.ndarray,
        *,
        max_windows: int = 0,
    ):
        self.dataset_path = dataset_path
        self.demo_keys = demo_keys
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.main_rgb_key = main_rgb_key
        self.sub_rgb_key = sub_rgb_key
        self.qpos_key = qpos_key
        self.action_key = action_key
        self.qpos_mean = torch.tensor(qpos_mean, dtype=torch.float32)
        self.qpos_std = torch.tensor(qpos_std, dtype=torch.float32)
        self._file: h5py.File | None = None
        self.index: list[tuple[str, int]] = []

        with h5py.File(dataset_path, "r") as f:
            for demo_key in demo_keys:
                demo = f[demo_key]
                num_steps = int(demo.attrs.get("num_steps", demo[action_key].shape[0]))
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        demo_key, end_idx = self.index[idx]
        demo = self._get_file()[demo_key]

        start_idx = max(0, end_idx - self.sequence_length + 1)
        pad = self.sequence_length - (end_idx - start_idx + 1)

        main_rgb = torch.from_numpy(strip_alpha_if_needed(demo[self.main_rgb_key][start_idx : end_idx + 1])).float()
        sub_rgb = torch.from_numpy(strip_alpha_if_needed(demo[self.sub_rgb_key][start_idx : end_idx + 1])).float()
        qpos = torch.from_numpy(demo[self.qpos_key][start_idx : end_idx + 1]).float()
        target_action = torch.from_numpy(demo[self.action_key][end_idx]).float()

        if pad > 0:
            main_rgb = torch.cat([main_rgb[:1].repeat(pad, 1, 1, 1), main_rgb], dim=0)
            sub_rgb = torch.cat([sub_rgb[:1].repeat(pad, 1, 1, 1), sub_rgb], dim=0)
            qpos = torch.cat([qpos[:1].repeat(pad, 1), qpos], dim=0)

        main_rgb = main_rgb.permute(0, 3, 1, 2) / 255.0
        sub_rgb = sub_rgb.permute(0, 3, 1, 2) / 255.0
        main_rgb = F.interpolate(main_rgb, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        sub_rgb = F.interpolate(sub_rgb, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        qpos = (qpos - self.qpos_mean) / self.qpos_std

        return {
            "main_rgb": main_rgb,
            "sub_rgb": sub_rgb,
            "qpos": qpos,
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
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
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


class QPosEncoder(nn.Module):
    def __init__(self, input_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, qpos: torch.Tensor) -> torch.Tensor:
        return self.net(qpos)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, model_dim = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float, mlp_ratio: float):
        super().__init__()
        hidden_dim = int(model_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = MultiHeadSelfAttention(model_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class ImageQposTransformerStudent(nn.Module):
    def __init__(
        self,
        qpos_dim: int,
        action_dim: int,
        sequence_length: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        mlp_ratio: float,
        camera_feature_dim: int,
        qpos_feature_dim: int,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.main_camera_encoder = CameraEncoder(camera_feature_dim)
        self.sub_camera_encoder = CameraEncoder(camera_feature_dim)
        self.qpos_encoder = QPosEncoder(qpos_dim, qpos_feature_dim)
        fusion_dim = camera_feature_dim * 2 + qpos_feature_dim
        self.input_proj = nn.Linear(fusion_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, sequence_length, model_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(model_dim, num_heads, dropout, mlp_ratio) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, action_dim)

    def forward(self, main_rgb: torch.Tensor, sub_rgb: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        main_feat = self.main_camera_encoder(main_rgb)
        sub_feat = self.sub_camera_encoder(sub_rgb)
        qpos_feat = self.qpos_encoder(qpos)
        x = torch.cat([main_feat, sub_feat, qpos_feat], dim=-1)
        x = self.input_proj(x)
        x = x + self.pos_embed[:, : x.shape[1]]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, -1])
        return self.head(x)


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
        qpos = batch["qpos"].to(device, non_blocking=True)
        target_action = batch["target_action"].to(device, non_blocking=True)

        pred_action = model(main_rgb, sub_rgb, qpos)
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
        model: ImageQposTransformerStudent,
        qpos_mean: torch.Tensor,
        qpos_std: torch.Tensor,
        image_size: int,
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.register_buffer("qpos_mean", qpos_mean)
        self.register_buffer("qpos_std", qpos_std)

    def forward(self, main_rgb: torch.Tensor, sub_rgb: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        main_rgb = main_rgb.float() / 255.0
        sub_rgb = sub_rgb.float() / 255.0
        if main_rgb.ndim == 4:
            main_rgb = main_rgb.unsqueeze(0)
            sub_rgb = sub_rgb.unsqueeze(0)
            qpos = qpos.unsqueeze(0)
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
        main_rgb = main_rgb.view(qpos.shape[0], qpos.shape[1], 3, main_rgb.shape[-2], main_rgb.shape[-1])
        sub_rgb = sub_rgb.view(qpos.shape[0], qpos.shape[1], 3, sub_rgb.shape[-2], sub_rgb.shape[-1])
        qpos = (qpos - self.qpos_mean) / self.qpos_std
        return self.model(main_rgb, sub_rgb, qpos)


def export_policy(model, stats: DatasetStats, output_dir: Path):
    export_wrapper = StudentPolicyExport(
        model,
        torch.tensor(stats.qpos_mean, dtype=torch.float32),
        torch.tensor(stats.qpos_std, dtype=torch.float32),
        stats.image_size,
    ).cpu()
    scripted = torch.jit.script(export_wrapper)
    scripted.save(str(output_dir / "policy.pt"))


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = resolve_output_dir(args.output_dir)
    print(f"[INFO] Output directory: {output_dir}")

    required_keys = [args.main_rgb_key, args.sub_rgb_key, args.qpos_key, args.action_key]
    demo_keys = list_demo_keys(args.dataset, required_keys)
    train_keys, val_keys = split_demo_keys(demo_keys, args.train_split, args.seed)
    qpos_mean, qpos_std, qpos_dim, action_dim = compute_qpos_normalization(
        args.dataset, train_keys, args.qpos_key, args.action_key
    )

    train_dataset = ImageQposDataset(
        args.dataset,
        train_keys,
        args.sequence_length,
        args.image_size,
        args.main_rgb_key,
        args.sub_rgb_key,
        args.qpos_key,
        args.action_key,
        qpos_mean,
        qpos_std,
        max_windows=args.max_train_windows,
    )
    val_dataset = ImageQposDataset(
        args.dataset,
        val_keys,
        args.sequence_length,
        args.image_size,
        args.main_rgb_key,
        args.sub_rgb_key,
        args.qpos_key,
        args.action_key,
        qpos_mean,
        qpos_std,
        max_windows=args.max_val_windows,
    )

    print(f"[INFO] Train demos: {len(train_keys)}, train windows: {len(train_dataset)}")
    print(f"[INFO] Val demos: {len(val_keys)}, val windows: {len(val_dataset)}")
    print(f"[INFO] qpos dim: {qpos_dim}, action dim: {action_dim}")

    train_loader = make_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_dataloader(val_dataset, args.batch_size, args.num_workers, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageQposTransformerStudent(
        qpos_dim=qpos_dim,
        action_dim=action_dim,
        sequence_length=args.sequence_length,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        camera_feature_dim=args.camera_feature_dim,
        qpos_feature_dim=args.qpos_feature_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    stats = DatasetStats(
        qpos_mean=qpos_mean.tolist(),
        qpos_std=qpos_std.tolist(),
        qpos_dim=qpos_dim,
        action_dim=action_dim,
        image_size=args.image_size,
        sequence_length=args.sequence_length,
    )
    with open(output_dir / "student_stats.json", "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)
    args_payload = vars(args).copy()
    args_payload["resolved_output_dir"] = str(output_dir)
    with open(output_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(args_payload, f, indent=2)

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
    print(f"[INFO] Exported scripted student policy to {output_dir / 'policy.pt'}")


if __name__ == "__main__":
    main()
