#!/usr/bin/env python3

"""Train a basic transformer student from HDF5 teacher trajectories.

Expected HDF5 layout:
  /demo_00000/policy_obs       [T, obs_dim]
  /demo_00000/policy_actions   [T, action_dim]
  /demo_00000 attrs["num_steps"] (optional)

The script trains a simple sequence model that predicts the teacher action at
the current step from the most recent observation window.
"""

from __future__ import annotations

import argparse
import json
import math
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
    parser = argparse.ArgumentParser(description="Train a basic transformer student from HDF5 teacher data.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 teacher dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for checkpoints and exports.")
    parser.add_argument("--sequence_length", type=int, default=16, help="Observation history length.")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--train_split", type=float, default=0.8, help="Episode split ratio.")
    parser.add_argument("--num_workers", type=int, default=0, help="PyTorch dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--model_dim", type=int, default=128, help="Transformer hidden size.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of transformer blocks.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="Feedforward hidden width ratio.")
    parser.add_argument("--max_train_windows", type=int, default=0, help="Optional cap on train windows.")
    parser.add_argument("--max_val_windows", type=int, default=0, help="Optional cap on val windows.")
    parser.add_argument("--obs_key", type=str, default="policy_obs", help="Observation dataset key inside each demo.")
    parser.add_argument(
        "--action_key", type=str, default="policy_actions", help="Teacher action dataset key inside each demo."
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DatasetStats:
    obs_mean: list[float]
    obs_std: list[float]
    obs_dim: int
    action_dim: int
    sequence_length: int


def list_demo_keys(dataset_path: str, obs_key: str, action_key: str) -> list[str]:
    with h5py.File(dataset_path, "r") as f:
        demo_keys = []
        for key in sorted(f.keys()):
            group = f[key]
            if isinstance(group, h5py.Group) and obs_key in group and action_key in group:
                demo_keys.append(key)
    if not demo_keys:
        raise ValueError(
            f"No demo groups with '{obs_key}' and '{action_key}' were found in dataset: {dataset_path}"
        )
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


def compute_normalization(
    dataset_path: str, demo_keys: list[str], obs_key: str, action_key: str
) -> tuple[np.ndarray, np.ndarray, int, int]:
    obs_sum = None
    obs_sq_sum = None
    total_steps = 0
    action_dim = 0

    with h5py.File(dataset_path, "r") as f:
        for demo_key in demo_keys:
            demo = f[demo_key]
            obs = demo[obs_key][:].astype(np.float32)
            actions = demo[action_key][:]
            if obs_sum is None:
                obs_sum = obs.sum(axis=0)
                obs_sq_sum = np.square(obs).sum(axis=0)
            else:
                obs_sum += obs.sum(axis=0)
                obs_sq_sum += np.square(obs).sum(axis=0)
            total_steps += obs.shape[0]
            action_dim = int(actions.shape[-1])

    mean = obs_sum / max(total_steps, 1)
    var = obs_sq_sum / max(total_steps, 1) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1.0e-6))
    return mean.astype(np.float32), std.astype(np.float32), int(mean.shape[0]), action_dim


class Hdf5PolicyDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        demo_keys: list[str],
        sequence_length: int,
        obs_key: str,
        action_key: str,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        *,
        max_windows: int = 0,
    ):
        self.dataset_path = dataset_path
        self.demo_keys = demo_keys
        self.sequence_length = sequence_length
        self.obs_key = obs_key
        self.action_key = action_key
        self.obs_mean = torch.tensor(obs_mean, dtype=torch.float32)
        self.obs_std = torch.tensor(obs_std, dtype=torch.float32)
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

        obs = torch.from_numpy(demo[self.obs_key][start_idx : end_idx + 1]).float()
        target_action = torch.from_numpy(demo[self.action_key][end_idx]).float()

        if pad > 0:
            obs = torch.cat([obs[:1].repeat(pad, 1), obs], dim=0)

        obs = (obs - self.obs_mean) / self.obs_std
        return {"obs": obs, "target_action": target_action}


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


class BasicTransformerStudent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        sequence_length: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        mlp_ratio: float,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.obs_embed = nn.Linear(obs_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, sequence_length, model_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(model_dim, num_heads, dropout, mlp_ratio) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.obs_embed(obs)
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
        obs = batch["obs"].to(device, non_blocking=True)
        target_action = batch["target_action"].to(device, non_blocking=True)

        pred_action = model(obs)
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
    def __init__(self, model: BasicTransformerStudent, obs_mean: torch.Tensor, obs_std: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_std", obs_std)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 2:
            obs = obs.unsqueeze(0)
        obs = (obs - self.obs_mean) / self.obs_std
        return self.model(obs)


def export_policy(model, stats: DatasetStats, output_dir: Path):
    export_wrapper = StudentPolicyExport(
        model,
        torch.tensor(stats.obs_mean, dtype=torch.float32),
        torch.tensor(stats.obs_std, dtype=torch.float32),
    ).cpu()
    scripted = torch.jit.script(export_wrapper)
    scripted.save(str(output_dir / "policy.pt"))


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_keys = list_demo_keys(args.dataset, args.obs_key, args.action_key)
    train_keys, val_keys = split_demo_keys(demo_keys, args.train_split, args.seed)
    obs_mean, obs_std, obs_dim, action_dim = compute_normalization(
        args.dataset, train_keys, args.obs_key, args.action_key
    )

    train_dataset = Hdf5PolicyDataset(
        args.dataset,
        train_keys,
        args.sequence_length,
        args.obs_key,
        args.action_key,
        obs_mean,
        obs_std,
        max_windows=args.max_train_windows,
    )
    val_dataset = Hdf5PolicyDataset(
        args.dataset,
        val_keys,
        args.sequence_length,
        args.obs_key,
        args.action_key,
        obs_mean,
        obs_std,
        max_windows=args.max_val_windows,
    )

    print(f"[INFO] Train demos: {len(train_keys)}, train windows: {len(train_dataset)}")
    print(f"[INFO] Val demos: {len(val_keys)}, val windows: {len(val_dataset)}")
    print(f"[INFO] Obs dim: {obs_dim}, action dim: {action_dim}")

    train_loader = make_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_dataloader(val_dataset, args.batch_size, args.num_workers, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicTransformerStudent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        sequence_length=args.sequence_length,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    stats = DatasetStats(
        obs_mean=obs_mean.tolist(),
        obs_std=obs_std.tolist(),
        obs_dim=obs_dim,
        action_dim=action_dim,
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
    print(f"[INFO] Exported scripted student policy to {output_dir / 'policy.pt'}")


if __name__ == "__main__":
    main()
