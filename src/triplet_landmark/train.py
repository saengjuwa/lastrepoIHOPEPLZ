from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from triplet_landmark.data import (
    MODEL_INPUT_SIZE,
    MODEL_MEAN,
    MODEL_STD,
    BalancedBatchSampler,
    canonical_path_key,
    ImageRecord,
    ImageTransform,
    LandmarkDataset,
    class_disjoint_split,
    country_disjoint_split,
    file_sha256,
    load_split_manifest,
    read_country_codes,
    read_korean_label_ids,
    read_label_csv,
    resolve_image_path,
    save_split_manifest,
)
from triplet_landmark.model import (
    MODEL_NAME,
    create_model,
    is_dinov2_backbone,
    resolve_backbone_name,
)
from triplet_landmark.predict_triplets import embed_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a configurable landmark embedding model."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/landmark_best.pt"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--backbone",
        choices=["efficientnetv2_s", "dinov2_small", "dinov2_base"],
        default="efficientnetv2_s",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Default: 300 for EfficientNet, 378 (14 x 27) for DINOv2.",
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--train-last-blocks", type=int, default=0)
    parser.add_argument("--use-projection", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--pooling", choices=["avg", "gem", "salad", "dolg"], default="avg")
    parser.add_argument("--gem-p", type=float, default=3.0)
    parser.add_argument("--salad-clusters", type=int, default=16)
    parser.add_argument("--salad-local-dim", type=int, default=64)
    parser.add_argument("--salad-global-dim", type=int, default=256)
    parser.add_argument("--sinkhorn-iterations", type=int, default=3)
    parser.add_argument("--dolg-dim", type=int, default=512)
    parser.add_argument(
        "--classifier",
        choices=["linear", "arcface", "subcenter_arcface"],
        default="linear",
    )
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.3)
    parser.add_argument("--subcenters", type=int, default=3)
    parser.add_argument(
        "--training-stage",
        choices=["classification", "metric"],
        default="metric",
    )
    parser.add_argument(
        "--metric-loss",
        choices=["none", "triplet", "supcon", "proxy_anchor"],
        default="triplet",
    )
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--proxy-alpha", type=float, default=32.0)
    parser.add_argument("--proxy-margin", type=float, default=0.1)
    parser.add_argument("--xbm-size", type=int, default=0)
    parser.add_argument("--xbm-weight", type=float, default=0.2)
    parser.add_argument("--xbm-warmup-steps", type=int, default=100)
    parser.add_argument("--augmentation", choices=["basic", "weak"], default="basic")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--init-checkpoint", type=Path, default=None)
    checkpoint_group.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-images-per-label",
        type=int,
        default=None,
        help="Default: 1 for classification stage, 2 for metric stage.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--triplet-weight", type=float, default=0.2)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--labels-per-batch", type=int, default=8)
    parser.add_argument("--images-per-label", type=int, default=2)
    # Kept for compatibility with old commands. Progress is now updated every batch.
    parser.add_argument("--log-every", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--select-best-triplet", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-val-triplets", type=int, default=1000)
    parser.add_argument("--hard-val-fraction", type=float, default=0.5)
    parser.add_argument(
        "--val-tta",
        choices=["none", "flip", "five_crop", "five_crop_flip"],
        default="flip",
    )
    parser.add_argument("--resize-mode", choices=["center_crop", "pad"], default="center_crop")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("outputs/landmark_split.json"),
    )
    parser.add_argument("--country-col", default="country_code")
    parser.add_argument("--korean-labels-file", type=Path, default=None)
    parser.add_argument("--val-countries-file", type=Path, default=None)
    parser.add_argument(
        "--data-audit-output",
        type=Path,
        default=Path("outputs/data_audit.json"),
    )
    parser.add_argument("--hard-negatives-csv", type=Path, default=None)
    parser.add_argument("--hard-negative-batch-size", type=int, default=32)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.7)
    parser.add_argument("--hard-negative-ratio-start", type=float, default=None)
    parser.add_argument("--hard-negative-weight", type=float, default=0.5)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    distance = 1.0 - embeddings @ embeddings.t()
    same_label = labels[:, None].eq(labels[None, :])
    eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    positive_mask = same_label & ~eye
    negative_mask = ~same_label
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not valid.any():
        return embeddings.new_tensor(0.0)

    hardest_positive = distance.masked_fill(~positive_mask, -1.0).max(dim=1).values
    hardest_negative = distance.masked_fill(~negative_mask, 2.0).min(dim=1).values
    return F.relu(
        hardest_positive[valid] - hardest_negative[valid] + margin
    ).mean()


DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)


def resolve_input_config(args: argparse.Namespace) -> None:
    args.model_name = resolve_backbone_name(args.backbone)
    if args.image_size is None:
        args.image_size = 378 if is_dinov2_backbone(args.model_name) else MODEL_INPUT_SIZE
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive.")
    if is_dinov2_backbone(args.model_name) and args.image_size % 14:
        raise ValueError("DINOv2 --image-size must be divisible by patch size 14.")
    args.input_mean = DINOV2_MEAN if is_dinov2_backbone(args.model_name) else MODEL_MEAN
    args.input_std = DINOV2_STD if is_dinov2_backbone(args.model_name) else MODEL_STD


def forward_for_training(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(model, "requires_labels_for_logits", False):
        return model(images, labels)
    return model(images)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("--supcon-temperature must be positive.")
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        stable_embeddings = embeddings.float()
        logits = stable_embeddings @ stable_embeddings.t() / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        self_mask = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
        denominator_mask = ~self_mask
        exp_logits = torch.exp(logits) * denominator_mask
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
        )
        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        if not valid.any():
            return stable_embeddings.new_tensor(0.0)
        return -(
            (log_prob * positive_mask).sum(dim=1)[valid] / positive_count[valid]
        ).mean()


class ProxyAnchorLoss(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        alpha: float,
        margin: float,
    ) -> None:
        super().__init__()
        if alpha <= 0 or margin < 0:
            raise ValueError("Proxy Anchor alpha must be positive and margin non-negative.")
        self.alpha = alpha
        self.margin = margin
        self.proxies = torch.nn.Parameter(torch.empty(num_classes, embedding_dim))
        torch.nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            stable_embeddings = embeddings.float()
            stable_proxies = F.normalize(self.proxies.float(), dim=1)
            similarities = stable_embeddings @ stable_proxies.t()
            positive_mask = F.one_hot(
                labels, num_classes=self.proxies.size(0)
            ).bool()
            negative_mask = ~positive_mask
            positive_sum = (
                torch.exp(-self.alpha * (similarities - self.margin))
                * positive_mask
            ).sum(dim=0)
            negative_sum = (
                torch.exp(self.alpha * (similarities + self.margin))
                * negative_mask
            ).sum(dim=0)
            positive_classes = positive_mask.any(dim=0)
            positive_term = torch.log1p(positive_sum[positive_classes]).mean()
            negative_term = torch.log1p(negative_sum).mean()
            return positive_term + negative_term


class CrossBatchMemory:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("XBM capacity must be positive.")
        self.capacity = capacity
        self.embeddings: torch.Tensor | None = None
        self.labels: torch.Tensor | None = None

    def __len__(self) -> int:
        return 0 if self.labels is None else self.labels.numel()

    def enqueue(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        new_embeddings = embeddings.detach().float().cpu()
        new_labels = labels.detach().long().cpu()
        if self.embeddings is None:
            combined_embeddings = new_embeddings
            combined_labels = new_labels
        else:
            combined_embeddings = torch.cat([self.embeddings, new_embeddings], dim=0)
            combined_labels = torch.cat([self.labels, new_labels], dim=0)
        self.embeddings = combined_embeddings[-self.capacity :]
        self.labels = combined_labels[-self.capacity :]

    def tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.embeddings is None or self.labels is None:
            raise ValueError("Cross-batch memory is empty.")
        return self.embeddings.to(device), self.labels.to(device)

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "embeddings": self.embeddings,
            "labels": self.labels,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("Resume checkpoint uses a different XBM capacity.")
        stored_embeddings = state.get("embeddings")
        stored_labels = state.get("labels")
        if (stored_embeddings is None) != (stored_labels is None):
            raise ValueError("XBM checkpoint has incomplete queue tensors.")
        if stored_embeddings is None:
            self.embeddings = None
            self.labels = None
            return
        if not isinstance(stored_embeddings, torch.Tensor) or not isinstance(
            stored_labels, torch.Tensor
        ):
            raise ValueError("XBM checkpoint queue values must be tensors.")
        if (
            stored_embeddings.ndim != 2
            or stored_labels.ndim != 1
            or stored_embeddings.size(0) != stored_labels.numel()
        ):
            raise ValueError("XBM checkpoint queue shapes do not match.")
        self.embeddings = (
            stored_embeddings.detach().float().cpu()[-self.capacity :]
        )
        self.labels = stored_labels.detach().long().cpu()[-self.capacity :]


def cross_batch_memory_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    memory_embeddings: torch.Tensor,
    memory_labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    distances = 1.0 - embeddings @ F.normalize(memory_embeddings, dim=1).t()
    positive_mask = labels[:, None].eq(memory_labels[None, :])
    negative_mask = ~positive_mask
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not valid.any():
        return embeddings.new_tensor(0.0)
    hardest_positive = distances.masked_fill(~positive_mask, -1.0).max(dim=1).values
    hardest_negative = distances.masked_fill(~negative_mask, 2.0).min(dim=1).values
    return F.relu(
        hardest_positive[valid] - hardest_negative[valid] + margin
    ).mean()


def selected_metric_loss(
    name: str,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    proxy_loss: ProxyAnchorLoss | None,
) -> torch.Tensor:
    if name == "none":
        return embeddings.new_tensor(0.0)
    if name == "triplet":
        return batch_hard_triplet_loss(embeddings, labels, args.triplet_margin)
    if name == "supcon":
        return supervised_contrastive_loss(
            embeddings, labels, args.supcon_temperature
        )
    if name == "proxy_anchor":
        if proxy_loss is None:
            raise RuntimeError("Proxy Anchor module was not created.")
        return proxy_loss(embeddings, labels)
    raise ValueError(f"Unsupported metric loss: {name}")


def hard_ratio_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    start = (
        args.hard_negative_ratio
        if args.hard_negative_ratio_start is None
        else args.hard_negative_ratio_start
    )
    if args.epochs <= 1:
        return args.hard_negative_ratio
    fraction = (epoch - 1) / (args.epochs - 1)
    return start + (args.hard_negative_ratio - start) * fraction


def load_initial_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    target_label_to_index: dict[str, int],
    args: argparse.Namespace,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    expected_config = {
        "model_name": args.model_name,
        "pooling": args.pooling,
        "use_projection": args.use_projection,
        "classifier_type": args.classifier,
    }
    if args.use_projection:
        expected_config["embedding_dim"] = args.embedding_dim
    if is_dinov2_backbone(args.model_name):
        expected_config["image_size"] = args.image_size
    if args.pooling == "salad":
        expected_config.update(
            {
                "salad_clusters": args.salad_clusters,
                "salad_local_dim": args.salad_local_dim,
                "salad_global_dim": args.salad_global_dim,
                "sinkhorn_iterations": args.sinkhorn_iterations,
            }
        )
    elif args.pooling == "dolg":
        expected_config["dolg_dim"] = args.dolg_dim
    defaults = {
        "model_name": MODEL_NAME,
        "pooling": "avg",
        "use_projection": False,
        "classifier_type": "linear",
        "salad_clusters": 16,
        "salad_local_dim": 64,
        "salad_global_dim": 256,
        "sinkhorn_iterations": 3,
        "dolg_dim": 512,
    }
    mismatches = {
        name: (checkpoint.get(name, defaults.get(name)), expected)
        for name, expected in expected_config.items()
        if checkpoint.get(name, defaults.get(name)) != expected
    }
    if mismatches:
        raise ValueError(
            f"Initial checkpoint configuration does not match: {mismatches}"
        )
    source_state = checkpoint.get("model_state", checkpoint)
    target_state = model.state_dict()
    source_label_to_index = checkpoint.get("label_to_index")
    labels_match = source_label_to_index == target_label_to_index
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state
        and target_state[key].shape == value.shape
        and (labels_match or not key.startswith("classifier."))
    }
    if not compatible:
        raise ValueError(f"No compatible weights found in {checkpoint_path}")
    target_state.update(compatible)
    model.load_state_dict(target_state)
    print(
        f"Loaded {len(compatible)} tensors from {checkpoint_path}; "
        f"skipped {len(source_state) - len(compatible)} "
        f"classifier_loaded={labels_match}",
        flush=True,
    )


class HardNegativeTripletDataset(Dataset):
    def __init__(
        self,
        triplets: list[tuple[Path, str, Path]],
        positives_by_label: dict[str, list[Path]],
        transform: ImageTransform,
        seed: int,
    ) -> None:
        self.triplets = triplets
        self.positives_by_label = positives_by_label
        self.transform = transform
        self.seed = seed

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_path, anchor_label, negative_path = self.triplets[index]
        positive_pool = self.positives_by_label[anchor_label]
        candidates = [path for path in positive_pool if path != anchor_path]
        if not candidates:
            candidates = positive_pool
        positive_path = random.choice(candidates)

        with Image.open(anchor_path) as anchor_image:
            anchor = self.transform(anchor_image)
        with Image.open(positive_path) as positive_image:
            positive = self.transform(positive_image)
        with Image.open(negative_path) as negative_image:
            negative = self.transform(negative_image)
        return anchor, positive, negative


def read_hard_negative_csv(
    csv_path: Path,
    image_root: Path,
    records: list[ImageRecord],
) -> list[tuple[Path, str, Path]]:
    label_by_path = {
        canonical_path_key(record.path): record.label for record in records
    }
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {
            "anchor_path",
            "anchor_label",
            "negative_path",
            "negative_label",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing hard-negative columns in {csv_path}: {sorted(missing)}")

        triplets = []
        seen: set[tuple[str, str]] = set()
        for row_number, row in enumerate(reader, start=2):
            anchor_label = row["anchor_label"]
            negative_label = row["negative_label"]
            anchor_path = resolve_image_path(image_root, row["anchor_path"])
            negative_path = resolve_image_path(image_root, row["negative_path"])
            anchor_key = canonical_path_key(anchor_path)
            negative_key = canonical_path_key(negative_path)
            actual_anchor_label = label_by_path.get(anchor_key)
            actual_negative_label = label_by_path.get(negative_key)
            if actual_anchor_label is None or actual_negative_label is None:
                raise ValueError(
                    f"Hard-negative row {row_number} contains a path outside the fixed "
                    "training split. Re-mine negatives using --split-manifest."
                )
            if actual_anchor_label != anchor_label:
                raise ValueError(
                    f"Hard-negative row {row_number} has anchor label {anchor_label!r}, "
                    f"but the training CSV says {actual_anchor_label!r}."
                )
            if actual_negative_label != negative_label:
                raise ValueError(
                    f"Hard-negative row {row_number} has negative label {negative_label!r}, "
                    f"but the training CSV says {actual_negative_label!r}."
                )
            if anchor_label == negative_label:
                raise ValueError(
                    f"Hard-negative row {row_number} uses the same label on both sides."
                )
            pair_key = (anchor_key, negative_key)
            if pair_key in seen:
                continue
            seen.add(pair_key)
            triplets.append((anchor_path, anchor_label, negative_path))

    if not triplets:
        raise ValueError(f"No usable hard negatives found in {csv_path}.")
    return triplets


def hard_negative_triplet_loss(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    positive_distance = 1.0 - (anchor_embeddings * positive_embeddings).sum(dim=1)
    negative_distance = 1.0 - (anchor_embeddings * negative_embeddings).sum(dim=1)
    return F.relu(positive_distance - negative_distance + margin).mean()


def make_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
    metric_module: torch.nn.Module | None = None,
) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)
    if metric_module is not None:
        head_params.extend(
            parameter for parameter in metric_module.parameters() if parameter.requires_grad
        )
    parameter_groups = []
    if backbone_params:
        parameter_groups.append({"params": backbone_params, "lr": args.backbone_lr})
    if head_params:
        parameter_groups.append({"params": head_params, "lr": args.lr})
    if not parameter_groups:
        raise ValueError("No trainable model parameters were found.")
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def make_checkpoint(
    model: torch.nn.Module,
    label_to_index: dict[str, int],
    args: argparse.Namespace,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    validation_metrics: dict[str, float] | None = None,
    best_accuracy: float | None = None,
    best_epoch: int | None = None,
    metric_module: torch.nn.Module | None = None,
    xbm: CrossBatchMemory | None = None,
) -> dict:
    image_size = getattr(args, "image_size", None) or MODEL_INPUT_SIZE
    model_name = getattr(args, "model_name", MODEL_NAME)
    input_mean = tuple(getattr(args, "input_mean", MODEL_MEAN))
    input_std = tuple(getattr(args, "input_std", MODEL_STD))
    return {
        "model_state": model.state_dict(),
        "metric_state": metric_module.state_dict() if metric_module is not None else None,
        "xbm_state": xbm.state_dict() if xbm is not None else None,
        "label_to_index": label_to_index,
        "model_name": model_name,
        "embedding_dim": model.embedding_dim,
        "image_size": image_size,
        "mean": input_mean,
        "std": input_std,
        "resize_mode": args.resize_mode,
        "pooling": args.pooling,
        "gem_p": args.gem_p,
        "use_projection": args.use_projection,
        "classifier_type": getattr(args, "classifier", "linear"),
        "arcface_scale": getattr(args, "arcface_scale", 30.0),
        "arcface_margin": getattr(args, "arcface_margin", 0.3),
        "subcenters": getattr(args, "subcenters", 3),
        "salad_clusters": getattr(args, "salad_clusters", 16),
        "salad_local_dim": getattr(args, "salad_local_dim", 64),
        "salad_global_dim": getattr(args, "salad_global_dim", 256),
        "sinkhorn_iterations": getattr(args, "sinkhorn_iterations", 3),
        "dolg_dim": getattr(args, "dolg_dim", 512),
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def resume_training_state(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    label_to_index: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    metric_module: torch.nn.Module | None = None,
    xbm: CrossBatchMemory | None = None,
) -> tuple[int, float, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("label_to_index") != label_to_index:
        raise ValueError("Resume checkpoint uses a different train/validation split.")
    saved_args = checkpoint.get("args", {})
    resume_argument_names = (
        "pooling", "gem_p", "use_projection", "embedding_dim", "resize_mode",
        "epochs", "batch_size", "labels_per_batch", "images_per_label", "lr",
        "backbone_lr", "weight_decay", "triplet_weight", "triplet_margin",
        "hard_negative_ratio", "hard_negative_weight",
        "hard_negatives_csv", "hard_negative_batch_size",
        "hard_negatives_sha256", "val_tta", "hard_val_fraction",
        "backbone", "image_size", "classifier",
        "metric_loss", "training_stage", "min_images_per_label", "augmentation", "xbm_size",
        "arcface_scale", "arcface_margin", "subcenters", "salad_clusters",
        "salad_local_dim", "salad_global_dim", "sinkhorn_iterations",
        "dolg_dim", "supcon_temperature", "proxy_alpha", "proxy_margin",
        "xbm_weight", "xbm_warmup_steps", "hard_negative_ratio_start",
        "freeze_backbone", "train_last_blocks",
    )
    for name in resume_argument_names:
        if name not in saved_args and not hasattr(args, name):
            continue
        saved_value = saved_args.get(name, getattr(args, name, None))
        current_value = getattr(args, name, saved_value)
        if name == "hard_negatives_csv":
            saved_value = (
                str(Path(saved_value).expanduser().resolve())
                if saved_value is not None
                else None
            )
            current_value = (
                str(Path(current_value).expanduser().resolve())
                if current_value is not None
                else None
            )
        if saved_value != current_value:
            raise ValueError(
                f"Resume argument mismatch for {name}: "
                f"checkpoint={saved_value!r} current={current_value!r}"
            )
    required = {"optimizer_state", "scheduler_state", "scaler_state", "epoch"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint cannot resume training; missing {sorted(missing)}")
    model.load_state_dict(checkpoint["model_state"])
    if metric_module is not None:
        metric_state = checkpoint.get("metric_state")
        if metric_state is None:
            raise ValueError("Resume checkpoint is missing Proxy Anchor state.")
        metric_module.load_state_dict(metric_state)
    if xbm is not None and checkpoint.get("xbm_state") is not None:
        xbm.load_state_dict(checkpoint["xbm_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_accuracy") or -1.0),
        int(checkpoint.get("best_epoch") or 0),
    )


def save_checkpoint(payload: dict, output: Path, epoch: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    epoch_output = output.with_name(f"{output.stem}_epoch{epoch:02d}{output.suffix}")
    torch.save(payload, epoch_output)
    torch.save(payload, output)
    print(f"Saved checkpoint: {epoch_output}", flush=True)


def make_validation_triplets(
    records: list[ImageRecord],
    max_triplets: int,
    seed: int,
    hard_negative_fraction: float,
    label_centroids: dict[str, torch.Tensor] | None,
) -> list[tuple[Path, Path, Path]]:
    if max_triplets <= 0:
        raise ValueError("--max-val-triplets must be positive.")
    if not 0.0 <= hard_negative_fraction <= 1.0:
        raise ValueError("--hard-val-fraction must be between 0 and 1.")
    if hard_negative_fraction > 0.0 and label_centroids is None:
        raise ValueError("Hard validation negatives require fixed label centroids.")

    paths_by_label_sets: dict[str, dict[str, Path]] = defaultdict(dict)
    for record in records:
        paths_by_label_sets[record.label][canonical_path_key(record.path)] = record.path
    paths_by_label = {
        label: list(paths_by_key.values())
        for label, paths_by_key in paths_by_label_sets.items()
    }
    labels = sorted(label for label, paths in paths_by_label.items() if len(paths) >= 2)
    if len(labels) < 2:
        raise ValueError("Triplet validation needs two labels with at least two images.")

    hard_negative_label: dict[str, str] = {}
    if label_centroids is not None:
        for label in labels:
            candidates = [candidate for candidate in labels if candidate != label]
            hard_negative_label[label] = max(
                candidates,
                key=lambda candidate: float(
                    torch.dot(label_centroids[label], label_centroids[candidate])
                ),
            )

    rng = random.Random(seed)
    hard_flags = [True] * round(max_triplets * hard_negative_fraction)
    hard_flags += [False] * (max_triplets - len(hard_flags))
    rng.shuffle(hard_flags)
    triplets: list[tuple[Path, Path, Path]] = []
    while len(triplets) < max_triplets:
        for label in labels:
            anchor, positive = rng.sample(paths_by_label[label], 2)
            if hard_flags[len(triplets)]:
                negative_label = hard_negative_label[label]
            else:
                negative_label = rng.choice(
                    [candidate for candidate in labels if candidate != label]
                )
            negative = rng.choice(paths_by_label[negative_label])
            triplets.append((anchor, positive, negative))
            if len(triplets) >= max_triplets:
                break
    return triplets


@torch.inference_mode()
def compute_validation_label_centroids(
    model: torch.nn.Module,
    records: list[ImageRecord],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    unique_paths = list(dict.fromkeys(record.path for record in records))
    model.eval()
    path_embeddings = embed_paths(
        paths=unique_paths,
        model=model,
        device=device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        tta=args.val_tta,
        mean=args.input_mean,
        std=args.input_std,
        resize_mode=args.resize_mode,
    )
    embeddings_by_label: dict[str, list[torch.Tensor]] = defaultdict(list)
    for record in records:
        embeddings_by_label[record.label].append(path_embeddings[str(record.path)])
    return {
        label: F.normalize(torch.stack(embeddings).mean(dim=0), p=2, dim=0)
        for label, embeddings in embeddings_by_label.items()
    }


@torch.inference_mode()
def evaluate_validation_triplets(
    model: torch.nn.Module,
    triplets: list[tuple[Path, Path, Path]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    unique_paths = list(dict.fromkeys(path for triplet in triplets for path in triplet))
    model.eval()
    path_embeddings = embed_paths(
        paths=unique_paths,
        model=model,
        device=device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        tta=args.val_tta,
        mean=args.input_mean,
        std=args.input_std,
        resize_mode=args.resize_mode,
    )
    path_to_embedding = {
        path: path_embeddings[str(path)] for path in unique_paths
    }

    sim_ap = torch.stack(
        [
            torch.dot(path_to_embedding[anchor], path_to_embedding[positive])
            for anchor, positive, _ in triplets
        ]
    )
    sim_an = torch.stack(
        [
            torch.dot(path_to_embedding[anchor], path_to_embedding[negative])
            for anchor, _, negative in triplets
        ]
    )
    return {
        "val_triplet_accuracy": float((sim_ap > sim_an).float().mean()),
        "mean_sim_ap": float(sim_ap.mean()),
        "mean_sim_an": float(sim_an.mean()),
        "mean_margin": float((sim_ap - sim_an).mean()),
    }


def print_batch_progress(
    epoch: int,
    step: int,
    total_steps: int,
    loss: float,
) -> None:
    percent = min(100.0, step / max(1, total_steps) * 100.0)
    bar_width = 30
    filled = round(bar_width * percent / 100.0)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(
        f"\repoch={epoch} [{bar}] {percent:6.2f}% "
        f"batch={step}/{total_steps} loss={loss:.5f}",
        end="",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    resolve_input_config(args)
    if args.min_images_per_label is None:
        args.min_images_per_label = (
            1 if args.training_stage == "classification" else 2
        )
    if args.min_images_per_label < 1:
        raise ValueError("--min-images-per-label must be positive.")
    if args.training_stage == "classification":
        if args.metric_loss != "none":
            raise ValueError("classification stage requires --metric-loss none.")
        if args.select_best_triplet:
            raise ValueError("classification stage cannot use triplet model selection.")
        if args.hard_negatives_csv is not None or args.xbm_size:
            raise ValueError("classification stage cannot use hard negatives or XBM.")
    else:
        if args.labels_per_batch * args.images_per_label != args.batch_size:
            raise ValueError(
                "--batch-size must equal --labels-per-batch * --images-per-label"
            )
        if args.metric_loss in {"triplet", "supcon"} and args.images_per_label < 2:
            raise ValueError(
                "--images-per-label must be at least 2 for triplet or SupCon loss."
            )
    if args.pooling == "salad" and not is_dinov2_backbone(args.model_name):
        raise ValueError("SALAD pooling requires a DINOv2 backbone.")
    if args.pooling == "dolg" and is_dinov2_backbone(args.model_name):
        raise ValueError("DOLG pooling currently supports CNN backbones only.")
    if args.train_last_blocks and not is_dinov2_backbone(args.model_name):
        raise ValueError("--train-last-blocks is only supported for DINOv2.")
    if args.xbm_size < 0 or args.xbm_weight < 0 or args.xbm_warmup_steps < 0:
        raise ValueError("XBM size, weight, and warm-up must be non-negative.")
    if not 0.0 <= args.hard_negative_ratio <= 1.0:
        raise ValueError("--hard-negative-ratio must be between 0 and 1.")
    if args.hard_negative_ratio_start is not None and not (
        0.0 <= args.hard_negative_ratio_start <= 1.0
    ):
        raise ValueError("--hard-negative-ratio-start must be between 0 and 1.")
    if args.hard_negative_batch_size <= 0:
        raise ValueError("--hard-negative-batch-size must be positive.")
    if args.hard_negative_weight < 0.0:
        raise ValueError("--hard-negative-weight must be non-negative.")
    if not 0.0 <= args.hard_val_fraction <= 1.0:
        raise ValueError("--hard-val-fraction must be between 0 and 1.")
    if args.resize_mode == "pad" and args.val_tta in {"five_crop", "five_crop_flip"}:
        raise ValueError(
            "five_crop validation TTA requires --resize-mode center_crop."
        )
    if args.select_best_triplet and args.hard_val_fraction > 0.0 and not (
        args.pretrained or args.init_checkpoint or args.resume_checkpoint
    ):
        raise ValueError(
            "Hard validation negatives need --pretrained, --init-checkpoint, "
            "or --resume-checkpoint so the fixed similarity ranking is meaningful."
        )

    args.hard_negatives_sha256 = (
        file_sha256(args.hard_negatives_csv)
        if args.hard_negatives_csv is not None
        else None
    )
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    korean_label_ids = (
        read_korean_label_ids(args.korean_labels_file)
        if args.korean_labels_file is not None
        else None
    )
    all_records, label_to_index, data_audit = read_label_csv(
        csv_path=args.csv,
        image_root=args.image_root,
        min_images_per_label=max(
            args.min_images_per_label,
            1 if args.training_stage == "classification" else 2,
        ),
        limit=args.limit,
        country_column=args.country_col,
        korean_label_ids=korean_label_ids,
    )
    args.data_audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_payload = data_audit.to_dict()
    audit_payload["csv"] = str(args.csv)
    audit_payload["csv_sha256"] = file_sha256(args.csv)
    args.data_audit_output.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved data audit: {args.data_audit_output}", flush=True)
    validation_triplets = None
    val_records: list[ImageRecord] | None = None
    if args.select_best_triplet:
        if args.split_manifest.exists():
            records, val_records, label_to_index = load_split_manifest(
                args.split_manifest,
                args.csv,
                all_records,
            )
            print(f"Loaded fixed split manifest: {args.split_manifest}", flush=True)
        else:
            val_countries = None
            split_strategy = "class_random"
            if args.val_countries_file is not None:
                val_countries = read_country_codes(args.val_countries_file)
                records, val_records, label_to_index = country_disjoint_split(
                    all_records,
                    val_countries,
                )
                split_strategy = "country_disjoint"
            else:
                records, val_records, label_to_index = class_disjoint_split(
                    all_records,
                    val_fraction=args.val_fraction,
                    seed=args.seed,
                )
            save_split_manifest(
                args.split_manifest,
                args.csv,
                records,
                val_records,
                args.seed,
                args.val_fraction,
                split_strategy,
                val_countries,
            )
            print(f"Saved fixed split manifest: {args.split_manifest}", flush=True)
    else:
        records = all_records
    if len(label_to_index) < 2:
        raise ValueError("Training requires at least two different landmark labels.")
    dataset = LandmarkDataset(
        records,
        ImageTransform(
            args.image_size,
            train=True,
            mean=args.input_mean,
            std=args.input_std,
            resize_mode=args.resize_mode,
            augmentation=args.augmentation,
        ),
    )
    train_transform = ImageTransform(
        args.image_size,
        train=True,
        mean=args.input_mean,
        std=args.input_std,
        resize_mode=args.resize_mode,
        augmentation=args.augmentation,
    )
    hard_loader = None
    hard_iterator = None
    if args.hard_negatives_csv is not None:
        positives_by_label: dict[str, list[Path]] = defaultdict(list)
        for record in records:
            positives_by_label[record.label].append(record.path)
        hard_triplets = read_hard_negative_csv(
            args.hard_negatives_csv,
            args.image_root,
            records,
        )
        hard_dataset = HardNegativeTripletDataset(
            hard_triplets,
            positives_by_label,
            train_transform,
            seed=args.seed,
        )
        hard_loader = DataLoader(
            hard_dataset,
            batch_size=args.hard_negative_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=len(hard_dataset) >= args.hard_negative_batch_size,
        )
        hard_iterator = iter(hard_loader)
    if args.training_stage == "classification":
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
        sampler = BalancedBatchSampler(
            records=records,
            labels_per_batch=args.labels_per_batch,
            images_per_label=args.images_per_label,
            seed=args.seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )


    model = create_model(
        num_classes=len(label_to_index),
        pretrained=args.pretrained,
        embedding_dim=args.embedding_dim,
        pooling=args.pooling,
        gem_p=args.gem_p,
        use_projection=args.use_projection,
        model_name=args.model_name,
        image_size=args.image_size,
        classifier_type=args.classifier,
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
        subcenters=args.subcenters,
        salad_clusters=args.salad_clusters,
        salad_local_dim=args.salad_local_dim,
        salad_global_dim=args.salad_global_dim,
        sinkhorn_iterations=args.sinkhorn_iterations,
        dolg_dim=args.dolg_dim,
    ).to(device)
    configure_backbone = getattr(model, "configure_backbone_training", None)
    if configure_backbone is not None:
        configure_backbone(args.freeze_backbone, args.train_last_blocks)

    if args.init_checkpoint is not None:
        load_initial_weights(
            model,
            args.init_checkpoint,
            device,
            label_to_index,
            args,
        )

    proxy_loss = (
        ProxyAnchorLoss(
            len(label_to_index),
            model.embedding_dim,
            args.proxy_alpha,
            args.proxy_margin,
        ).to(device)
        if args.metric_loss == "proxy_anchor"
        else None
    )
    xbm = CrossBatchMemory(args.xbm_size) if args.xbm_size else None
    optimizer = make_optimizer(model, args, proxy_loss)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1
    best_accuracy = -1.0
    best_epoch = 0
    if args.resume_checkpoint is not None:
        start_epoch, best_accuracy, best_epoch = resume_training_state(
            args.resume_checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
            label_to_index,
            args,
            device,
            proxy_loss,
            xbm,
        )
        if start_epoch > args.epochs:
            raise ValueError(
                f"Resume checkpoint already reached epoch {start_epoch - 1}; "
                f"--epochs is {args.epochs}."
            )
        print(f"Resuming from epoch {start_epoch}: {args.resume_checkpoint}", flush=True)

    if val_records is not None:
        label_centroids = None
        if args.hard_val_fraction > 0.0:
            label_centroids = compute_validation_label_centroids(
                model,
                val_records,
                args,
                device,
            )
        validation_triplets = make_validation_triplets(
            val_records,
            max_triplets=args.max_val_triplets,
            seed=args.seed,
            hard_negative_fraction=args.hard_val_fraction,
            label_centroids=label_centroids,
        )

    print(
        f"model={args.model_name} images={len(records)} labels={len(label_to_index)} "
        f"input={args.image_size} batch={args.batch_size} device={device} "
        f"device_name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'} "
        f"pooling={args.pooling} classifier={args.classifier} "
        f"metric_loss={args.metric_loss} use_projection={args.use_projection} "
        f"resize_mode={args.resize_mode} augmentation={args.augmentation} "
        f"val_tta={args.val_tta} "
        f"hard_negatives={args.hard_negatives_csv is not None}",
        flush=True,
    )
    last_output = args.output.with_name(f"{args.output.stem}_last{args.output.suffix}")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if proxy_loss is not None:
            proxy_loss.train()
        total_loss = 0.0
        total_ce = 0.0
        total_metric = 0.0
        total_xbm = 0.0
        total_hard = 0.0
        total_seen = 0
        hard_seen = 0
        current_hard_ratio = hard_ratio_for_epoch(args, epoch)
        for step, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                embeddings, logits = forward_for_training(model, images, labels)
                ce_loss = F.cross_entropy(logits, labels)
                metric_loss = selected_metric_loss(
                    args.metric_loss,
                    embeddings,
                    labels,
                    args,
                    proxy_loss,
                )
                xbm_loss = embeddings.new_tensor(0.0)
                if (
                    xbm is not None
                    and len(xbm)
                    and (epoch - 1) * len(loader) + step > args.xbm_warmup_steps
                ):
                    memory_embeddings, memory_labels = xbm.tensors(device)
                    xbm_loss = cross_batch_memory_loss(
                        embeddings,
                        labels,
                        memory_embeddings,
                        memory_labels,
                        args.triplet_margin,
                    )
                hard_loss = embeddings.new_tensor(0.0)
                use_hard_batch = (
                    hard_loader is not None
                    and hard_iterator is not None
                    and random.random() < current_hard_ratio
                )
                if use_hard_batch:
                    try:
                        hard_anchor, hard_positive, hard_negative = next(hard_iterator)
                    except StopIteration:
                        hard_iterator = iter(hard_loader)
                        hard_anchor, hard_positive, hard_negative = next(hard_iterator)
                    hard_images = torch.cat(
                        [hard_anchor, hard_positive, hard_negative],
                        dim=0,
                    ).to(device, non_blocking=True)
                    hard_embeddings, _ = model(hard_images)
                    anchor_embeddings, positive_embeddings, negative_embeddings = (
                        hard_embeddings.chunk(3, dim=0)
                    )
                    hard_loss = hard_negative_triplet_loss(
                        anchor_embeddings,
                        positive_embeddings,
                        negative_embeddings,
                        args.triplet_margin,
                    )
                    hard_batch_size = hard_anchor.size(0)
                    total_hard += hard_loss.item() * hard_batch_size
                    hard_seen += hard_batch_size
                loss = (
                    ce_loss
                    + args.triplet_weight * metric_loss
                    + args.xbm_weight * xbm_loss
                    + args.hard_negative_weight * hard_loss
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if xbm is not None:
                xbm.enqueue(embeddings, labels)

            seen = images.size(0)
            total_loss += loss.item() * seen
            total_ce += ce_loss.item() * seen
            total_metric += metric_loss.item() * seen
            total_xbm += xbm_loss.item() * seen
            total_seen += seen
            print_batch_progress(epoch, step, len(loader), loss.item())

        print()
        scheduler.step()
        mean_hard_loss = total_hard / hard_seen if hard_seen else 0.0
        print(
            f"epoch={epoch} loss={total_loss / total_seen:.5f} "
            f"ce={total_ce / total_seen:.5f} "
            f"metric={total_metric / total_seen:.5f} "
            f"xbm={total_xbm / total_seen:.5f} "
            f"hard={mean_hard_loss:.5f} "
            f"hard_ratio={current_hard_ratio:.3f} hard_samples={hard_seen}",
            flush=True,
        )

        if validation_triplets is None:
            save_checkpoint(
                make_checkpoint(
                    model,
                    label_to_index,
                    args,
                    epoch,
                    optimizer,
                    scheduler,
                    scaler,
                    metric_module=proxy_loss,
                    xbm=xbm,
                ),
                args.output,
                epoch,
            )
            continue

        metrics = evaluate_validation_triplets(model, validation_triplets, args, device)
        improved = metrics["val_triplet_accuracy"] > best_accuracy
        if improved:
            best_accuracy = metrics["val_triplet_accuracy"]
            best_epoch = epoch
        payload = make_checkpoint(
            model,
            label_to_index,
            args,
            epoch,
            optimizer,
            scheduler,
            scaler,
            metrics,
            best_accuracy,
            best_epoch,
            metric_module=proxy_loss,
            xbm=xbm,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, last_output)
        if improved:
            torch.save(payload, args.output)
        print(
            f"epoch={epoch} val_triplet_accuracy="
            f"{metrics['val_triplet_accuracy']:.6f} "
            f"mean_sim_ap={metrics['mean_sim_ap']:.6f} "
            f"mean_sim_an={metrics['mean_sim_an']:.6f} "
            f"mean_margin={metrics['mean_margin']:.6f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )

    if validation_triplets is not None:
        print(
            f"Selected best triplet checkpoint: {args.output} "
            f"(epoch={best_epoch}, accuracy={best_accuracy:.6f})",
            flush=True,
        )
        print(f"Saved last epoch checkpoint: {last_output}", flush=True)


if __name__ == "__main__":
    main()
