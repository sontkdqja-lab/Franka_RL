#!/usr/bin/env python3

"""Train an image+qpos behavior cloning student with ResNet-18 encoders and an MLP head."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ResNet18 + qpos MLP behavior cloning policy from HDF5 demonstrations."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 teacher dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for checkpoints and exports.")
    parser.add_argument("--image_size", type=int, default=128, help="Square image size after resizing.")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--train_split", type=float, default=0.8, help="Episode split ratio.")
    parser.add_argument("--num_workers", type=int, default=0, help="PyTorch dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--image_feature_dim", type=int, default=128, help="Projected latent size for each camera encoder."
    )
    parser.add_argument(
        "--mlp_hidden_dims",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Hidden layer sizes for the policy MLP.",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability for the policy MLP.")
    parser.add_argument(
        "--disable_sub_camera",
        action="store_true",
        help="Ignore the secondary camera and train from main_rgb + qpos only.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze the ResNet-18 backbone and train only the projection and policy head.",
    )
    parser.add_argument(
        "--disable_pretrained_backbone",
        action="store_true",
        help="Disable ImageNet-pretrained ResNet-18 weights and train the visual backbone from scratch.",
    )
    parser.add_argument("--max_train_steps", type=int, default=0, help="Optional cap on train samples.")
    parser.add_argument("--max_val_steps", type=int, default=0, help="Optional cap on val samples.")
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
    qpos_dim: int
    action_dim: int
    image_size: int
    use_sub_camera: bool
    qpos_mean: list[float]
    qpos_std: list[float]


def strip_alpha_if_needed(images: np.ndarray) -> np.ndarray:
    if images.shape[-1] == 4:
        return images[..., :3]
    return images


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


def infer_dims(dataset_path: str, demo_keys: list[str], qpos_key: str, action_key: str) -> tuple[int, int]:
    with h5py.File(dataset_path, "r") as f:
        first_demo = f[demo_keys[0]]
        qpos_dim = int(first_demo[qpos_key].shape[-1])
        action_dim = int(first_demo[action_key].shape[-1])
    return qpos_dim, action_dim


def compute_qpos_stats(dataset_path: str, index: list[tuple[str, int]], qpos_key: str) -> tuple[np.ndarray, np.ndarray]:
    if not index:
        raise ValueError("Cannot compute qpos statistics from an empty dataset index.")

    with h5py.File(dataset_path, "r") as f:
        qpos_dim = int(f[index[0][0]][qpos_key].shape[-1])
        qpos_sum = np.zeros(qpos_dim, dtype=np.float64)
        qpos_sq_sum = np.zeros(qpos_dim, dtype=np.float64)

        for demo_key, step_idx in index:
            qpos = np.asarray(f[demo_key][qpos_key][step_idx], dtype=np.float64)
            qpos_sum += qpos
            qpos_sq_sum += np.square(qpos)

    count = float(len(index))
    mean = qpos_sum / count
    var = np.maximum(qpos_sq_sum / count - np.square(mean), 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


class ImageQposStepDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        demo_keys: list[str],
        image_size: int,
        main_rgb_key: str,
        sub_rgb_key: str,
        qpos_key: str,
        action_key: str,
        qpos_mean: np.ndarray,
        qpos_std: np.ndarray,
        *,
        use_sub_camera: bool,
        max_steps: int = 0,
    ):
        self.dataset_path = dataset_path
        self.demo_keys = demo_keys
        self.image_size = image_size
        self.main_rgb_key = main_rgb_key
        self.sub_rgb_key = sub_rgb_key
        self.qpos_key = qpos_key
        self.action_key = action_key
        self.use_sub_camera = use_sub_camera
        self.qpos_mean = torch.as_tensor(qpos_mean, dtype=torch.float32)
        self.qpos_std = torch.as_tensor(qpos_std, dtype=torch.float32)
        self._file: h5py.File | None = None
        self.index: list[tuple[str, int]] = []

        with h5py.File(dataset_path, "r") as f:
            for demo_key in demo_keys:
                demo = f[demo_key]
                num_steps = int(demo.attrs.get("num_steps", demo[action_key].shape[0]))
                for step_idx in range(num_steps):
                    self.index.append((demo_key, step_idx))

        if max_steps > 0 and len(self.index) > max_steps:
            self.index = self.index[:max_steps]

    def __len__(self) -> int:
        return len(self.index)

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.dataset_path, "r")
        return self._file

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        demo_key, step_idx = self.index[idx]
        demo = self._get_file()[demo_key]

        main_rgb = torch.from_numpy(strip_alpha_if_needed(demo[self.main_rgb_key][step_idx])).float()
        qpos = torch.from_numpy(demo[self.qpos_key][step_idx]).float()
        qpos = (qpos - self.qpos_mean) / self.qpos_std
        target_action = torch.from_numpy(demo[self.action_key][step_idx]).float()

        main_rgb = main_rgb.permute(2, 0, 1) / 255.0
        main_rgb = F.interpolate(
            main_rgb.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        sample = {
            "main_rgb": main_rgb,
            "qpos": qpos,
            "target_action": target_action,
        }

        if self.use_sub_camera:
            sub_rgb = torch.from_numpy(strip_alpha_if_needed(demo[self.sub_rgb_key][step_idx])).float()
            sub_rgb = sub_rgb.permute(2, 0, 1) / 255.0
            sub_rgb = F.interpolate(
                sub_rgb.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            sample["sub_rgb"] = sub_rgb

        return sample


class ResNet18Encoder(nn.Module):
    def __init__(self, out_dim: int, freeze_backbone: bool, pretrained_backbone: bool):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(backbone.fc.in_features, out_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.mean) / self.std
        x = self.backbone(images).flatten(1)
        return self.proj(x)


class ImageQposResNetMLPPolicy(nn.Module):
    def __init__(
        self,
        qpos_dim: int,
        action_dim: int,
        image_feature_dim: int,
        mlp_hidden_dims: list[int],
        dropout: float,
        use_sub_camera: bool,
        freeze_backbone: bool,
        pretrained_backbone: bool,
    ):
        super().__init__()
        self.use_sub_camera = use_sub_camera
        self.main_encoder = ResNet18Encoder(
            image_feature_dim,
            freeze_backbone=freeze_backbone,
            pretrained_backbone=pretrained_backbone,
        )
        self.sub_encoder = (
            ResNet18Encoder(
                image_feature_dim,
                freeze_backbone=freeze_backbone,
                pretrained_backbone=pretrained_backbone,
            )
            if use_sub_camera
            else None
        )

        input_dim = image_feature_dim + qpos_dim
        if use_sub_camera:
            input_dim += image_feature_dim

        mlp_layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, action_dim))
        self.policy_head = nn.Sequential(*mlp_layers)

    def forward(self, main_rgb: torch.Tensor, qpos: torch.Tensor, sub_rgb: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = [self.main_encoder(main_rgb), qpos]
        if self.use_sub_camera:
            if sub_rgb is None:
                raise ValueError("sub_rgb must be provided when use_sub_camera=True.")
            features.insert(1, self.sub_encoder(sub_rgb))
        fused = torch.cat(features, dim=-1)
        return self.policy_head(fused)


def make_dataloader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def run_epoch(model, loader, optimizer, device, use_sub_camera: bool):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        main_rgb = batch["main_rgb"].to(device, non_blocking=True)
        qpos = batch["qpos"].to(device, non_blocking=True)
        target_action = batch["target_action"].to(device, non_blocking=True)
        sub_rgb = batch["sub_rgb"].to(device, non_blocking=True) if use_sub_camera else None

        pred_action = model(main_rgb, qpos, sub_rgb)
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
        model: ImageQposResNetMLPPolicy,
        image_size: int,
        use_sub_camera: bool,
        qpos_mean: list[float],
        qpos_std: list[float],
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.use_sub_camera = use_sub_camera
        self.register_buffer("qpos_mean", torch.tensor(qpos_mean, dtype=torch.float32))
        self.register_buffer("qpos_std", torch.tensor(qpos_std, dtype=torch.float32))

    def _preprocess(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.float() / 255.0
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        rgb = rgb.permute(0, 3, 1, 2)
        rgb = F.interpolate(rgb, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return rgb

    def forward(
        self, main_rgb: torch.Tensor, qpos: torch.Tensor, sub_rgb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        main_rgb = self._preprocess(main_rgb)
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        qpos = (qpos.float() - self.qpos_mean) / self.qpos_std
        if self.use_sub_camera:
            if sub_rgb is None:
                raise ValueError("sub_rgb must be provided when use_sub_camera=True.")
            sub_rgb = self._preprocess(sub_rgb)
        return self.model(main_rgb, qpos, sub_rgb)


def export_policy(model, stats: DatasetStats, output_dir: Path):
    export_wrapper = StudentPolicyExport(
        model,
        image_size=stats.image_size,
        use_sub_camera=stats.use_sub_camera,
        qpos_mean=stats.qpos_mean,
        qpos_std=stats.qpos_std,
    ).cpu()
    scripted = torch.jit.script(export_wrapper)
    scripted.save(str(output_dir / "policy.pt"))


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = resolve_output_dir(args.output_dir)
    print(f"[INFO] Output directory: {output_dir}")

    required_keys = [args.main_rgb_key, args.qpos_key, args.action_key]
    use_sub_camera = not args.disable_sub_camera
    if use_sub_camera:
        required_keys.append(args.sub_rgb_key)

    demo_keys = list_demo_keys(args.dataset, required_keys)
    train_keys, val_keys = split_demo_keys(demo_keys, args.train_split, args.seed)
    qpos_dim, action_dim = infer_dims(args.dataset, train_keys, args.qpos_key, args.action_key)

    train_index: list[tuple[str, int]] = []
    with h5py.File(args.dataset, "r") as f:
        for demo_key in train_keys:
            demo = f[demo_key]
            num_steps = int(demo.attrs.get("num_steps", demo[args.action_key].shape[0]))
            for step_idx in range(num_steps):
                train_index.append((demo_key, step_idx))
    if args.max_train_steps > 0 and len(train_index) > args.max_train_steps:
        train_index = train_index[: args.max_train_steps]
    qpos_mean, qpos_std = compute_qpos_stats(args.dataset, train_index, args.qpos_key)

    train_dataset = ImageQposStepDataset(
        args.dataset,
        train_keys,
        args.image_size,
        args.main_rgb_key,
        args.sub_rgb_key,
        args.qpos_key,
        args.action_key,
        qpos_mean=qpos_mean,
        qpos_std=qpos_std,
        use_sub_camera=use_sub_camera,
        max_steps=args.max_train_steps,
    )
    val_dataset = ImageQposStepDataset(
        args.dataset,
        val_keys,
        args.image_size,
        args.main_rgb_key,
        args.sub_rgb_key,
        args.qpos_key,
        args.action_key,
        qpos_mean=qpos_mean,
        qpos_std=qpos_std,
        use_sub_camera=use_sub_camera,
        max_steps=args.max_val_steps,
    )

    print(f"[INFO] Train demos: {len(train_keys)}, train steps: {len(train_dataset)}")
    print(f"[INFO] Val demos: {len(val_keys)}, val steps: {len(val_dataset)}")
    print(f"[INFO] qpos dim: {qpos_dim}, action dim: {action_dim}, use_sub_camera: {use_sub_camera}")
    print(
        f"[INFO] qpos normalization ready: mean shape={qpos_mean.shape}, std range=({qpos_std.min():.6f}, {qpos_std.max():.6f})"
    )

    train_loader = make_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_dataloader(val_dataset, args.batch_size, args.num_workers, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageQposResNetMLPPolicy(
        qpos_dim=qpos_dim,
        action_dim=action_dim,
        image_feature_dim=args.image_feature_dim,
        mlp_hidden_dims=args.mlp_hidden_dims,
        dropout=args.dropout,
        use_sub_camera=use_sub_camera,
        freeze_backbone=args.freeze_backbone,
        pretrained_backbone=not args.disable_pretrained_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    stats = DatasetStats(
        qpos_dim=qpos_dim,
        action_dim=action_dim,
        image_size=args.image_size,
        use_sub_camera=use_sub_camera,
        qpos_mean=qpos_mean.tolist(),
        qpos_std=qpos_std.tolist(),
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
        train_loss = run_epoch(model, train_loader, optimizer, device, use_sub_camera)
        val_loss = run_epoch(model, val_loader, None, device, use_sub_camera)
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
