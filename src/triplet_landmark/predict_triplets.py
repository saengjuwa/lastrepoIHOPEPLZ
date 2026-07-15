from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from pathlib import Path

import torch
from PIL import Image

from torch.nn import functional as F

from triplet_landmark.data import (
    MODEL_INPUT_SIZE,
    MODEL_MEAN,
    MODEL_STD,
    make_inference_tensors,
    resolve_image_path,
)
from triplet_landmark.model import MODEL_NAME, create_model


TRIPLET_COLUMN_CANDIDATES = [
    ("anchor", "positive", "negative"),
    ("anchor_path", "positive_path", "negative_path"),
    ("A1", "A2", "B1"),
    ("a1", "a2", "b1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write cosine similarities for triplet images.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, nargs="+")
    source.add_argument("--embedding-db", type=Path)
    parser.add_argument("--triplets", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/triplet_scores.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--tta",
        choices=["none", "flip", "five_crop", "five_crop_flip"],
        default="none",
    )
    parser.add_argument(
        "--scales",
        default="",
        help="Comma-separated inference sizes. Empty uses each checkpoint's training size.",
    )
    parser.add_argument(
        "--local-reranker",
        choices=["none", "lightglue"],
        default="none",
    )
    parser.add_argument("--local-weight", type=float, default=0.05)
    parser.add_argument("--local-margin-threshold", type=float, default=0.05)
    parser.add_argument(
        "--local-features",
        choices=["aliked", "disk", "superpoint"],
        default="aliked",
    )
    parser.add_argument("--save-embedding-db", type=Path, default=None)
    parser.add_argument("--anchor-col", default=None)
    parser.add_argument("--positive-col", default=None)
    parser.add_argument("--negative-col", default=None)
    return parser.parse_args()


def parse_scales(value: str) -> list[int] | None:
    if not value.strip():
        return None
    try:
        scales = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--scales must be comma-separated positive integers.") from exc
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("--scales must contain positive integers.")
    return list(dict.fromkeys(scales))


def resolve_columns(
    fieldnames: list[str],
    anchor_col: str | None,
    positive_col: str | None,
    negative_col: str | None,
) -> tuple[str, str, str]:
    if anchor_col and positive_col and negative_col:
        requested = [anchor_col, positive_col, negative_col]
        missing = [column for column in requested if column not in fieldnames]
        if missing:
            raise ValueError(f"Missing requested triplet columns: {missing}")
        return anchor_col, positive_col, negative_col

    for columns in TRIPLET_COLUMN_CANDIDATES:
        if all(column in fieldnames for column in columns):
            return columns
    raise ValueError(
        "Could not infer triplet columns. Pass --anchor-col, --positive-col, and --negative-col."
    )


def read_triplet_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8-sig") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "triplets" in payload:
            payload = payload["triplets"]
        elif isinstance(payload, dict) and {"anchor", "positive", "negative"} <= set(payload):
            payload = [payload]
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise ValueError("Triplet JSON must be a list of objects or an object with a triplets list.")

        fieldnames: list[str] = []
        rows: list[dict[str, str]] = []
        for raw_row in payload:
            row = {str(key): str(value) for key, value in raw_row.items()}
            rows.append(row)
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        return rows, fieldnames

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [{key: value for key, value in row.items()} for row in reader]
        return rows, list(reader.fieldnames or [])


def _normalized_strings(value: str) -> list[str]:
    values: list[str] = []
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        normalized = unicodedata.normalize(form, value)
        if normalized not in values:
            values.append(normalized)
    return values


def resolve_existing_image_path(image_root: Path, value: str) -> Path:
    requested = resolve_image_path(image_root, value)
    if requested.exists():
        return requested

    for candidate_value in _normalized_strings(str(requested)):
        candidate = Path(candidate_value)
        if candidate.exists():
            return candidate

    parent = requested.parent
    if parent.exists():
        requested_names = set(_normalized_strings(requested.name))
        for existing in parent.iterdir():
            if requested_names.intersection(_normalized_strings(existing.name)):
                return existing

    raise FileNotFoundError(
        "Image file not found. "
        f"requested={requested} image_root={image_root} triplet_value={value}"
    )


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_to_index = checkpoint["label_to_index"]
    model_name = str(checkpoint.get("model_name", MODEL_NAME))
    image_size = int(checkpoint.get("image_size", MODEL_INPUT_SIZE))
    model = create_model(
        num_classes=len(label_to_index),
        pretrained=False,
        embedding_dim=int(checkpoint.get("embedding_dim", 512)),
        pooling=checkpoint.get("pooling", "avg"),
        gem_p=float(checkpoint.get("gem_p", 3.0)),
        use_projection=bool(checkpoint.get("use_projection", False)),
        model_name=model_name,
        image_size=image_size,
        classifier_type=checkpoint.get("classifier_type", "linear"),
        arcface_scale=float(checkpoint.get("arcface_scale", 30.0)),
        arcface_margin=float(checkpoint.get("arcface_margin", 0.3)),
        subcenters=int(checkpoint.get("subcenters", 3)),
        salad_clusters=int(checkpoint.get("salad_clusters", 16)),
        salad_local_dim=int(checkpoint.get("salad_local_dim", 64)),
        salad_global_dim=int(checkpoint.get("salad_global_dim", 256)),
        sinkhorn_iterations=int(checkpoint.get("sinkhorn_iterations", 3)),
        dolg_dim=int(checkpoint.get("dolg_dim", 512)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    mean = tuple(checkpoint.get("mean", MODEL_MEAN))
    std = tuple(checkpoint.get("std", MODEL_STD))
    resize_mode = str(checkpoint.get("resize_mode", "center_crop"))
    return model, image_size, mean, std, resize_mode


@torch.inference_mode()
def embed_path_matrix(
    paths: list[Path],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    resize_mode: str,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    chunks: list[torch.Tensor] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(
                    make_inference_tensors(
                        image,
                        image_size,
                        tta="none",
                        mean=mean,
                        std=std,
                        resize_mode=resize_mode,
                    )[0]
                )
        batch = torch.stack(images).to(device)
        batch_embeddings, _ = model(batch)
        chunks.append(batch_embeddings.float().cpu())
    if not chunks:
        return torch.empty((0, int(getattr(model, "embedding_dim", 0))))
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def _embed_paths_single_scale(
    paths: list[Path],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,
    tta: str,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    resize_mode: str = "center_crop",
) -> dict[str, torch.Tensor]:
    embeddings: dict[str, torch.Tensor] = {}
    if tta == "none":
        matrix = embed_path_matrix(
            paths,
            model,
            device,
            batch_size,
            image_size,
            mean,
            std,
            resize_mode,
        )
        for path, embedding in zip(paths, matrix):
            embeddings[str(path)] = embedding
        return embeddings

    for path in paths:
        with Image.open(path) as image:
            tensors = make_inference_tensors(
                image,
                image_size,
                tta=tta,
                mean=mean,
                std=std,
                resize_mode=resize_mode,
            )
        batch_embeddings = []
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size]).to(device)
            embedding, _ = model(batch)
            batch_embeddings.append(embedding.cpu())
        embedding = torch.cat(batch_embeddings, dim=0).mean(dim=0)
        embeddings[str(path)] = F.normalize(embedding, p=2, dim=0)
    return embeddings


@torch.inference_mode()
def embed_paths(
    paths: list[Path],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,
    tta: str,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    resize_mode: str = "center_crop",
    scales: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    selected_scales = scales or [image_size]
    if any(scale <= 0 for scale in selected_scales):
        raise ValueError("Inference scales must be positive.")
    if "dinov2" in str(getattr(model, "model_name", "")).lower() and any(
        scale % 14 for scale in selected_scales
    ):
        raise ValueError("DINOv2 inference scales must be divisible by patch size 14.")
    accumulated: dict[str, list[torch.Tensor]] = {str(path): [] for path in paths}
    for scale in selected_scales:
        scale_embeddings = _embed_paths_single_scale(
            paths=paths,
            model=model,
            device=device,
            batch_size=batch_size,
            image_size=scale,
            tta=tta,
            mean=mean,
            std=std,
            resize_mode=resize_mode,
        )
        for key, embedding in scale_embeddings.items():
            accumulated[key].append(embedding)
    return {
        key: F.normalize(torch.stack(values).mean(dim=0), p=2, dim=0)
        for key, values in accumulated.items()
    }


def load_embedding_db(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"Unsupported embedding DB format: {path}")
    values = payload["values"]
    embeddings = payload["embeddings"]
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError(f"Embedding DB must contain a 2-D tensor: {path}")
    if len(values) != len(embeddings):
        raise ValueError(f"Embedding DB value and embedding counts differ: {path}")
    if len(set(map(str, values))) != len(values):
        raise ValueError(f"Embedding DB contains duplicate keys: {path}")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"Embedding DB contains non-finite values: {path}")
    if len(embeddings):
        norms = embeddings.float().norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3, rtol=1e-3):
            raise ValueError(f"Embedding DB contains non-normalized embeddings: {path}")
    return {str(value): embedding for value, embedding in zip(values, embeddings)}


def save_embedding_db(
    path: Path,
    value_to_embedding: dict[str, torch.Tensor],
    checkpoints: list[Path],
    tta: str,
    resize_mode: str,
    scales: list[int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(value_to_embedding)
    embeddings = torch.stack([value_to_embedding[value].cpu() for value in values])
    torch.save(
        {
            "format_version": 1,
            "values": values,
            "embeddings": embeddings,
            "checkpoints": [str(checkpoint) for checkpoint in checkpoints],
            "tta": tta,
            "scales": scales,
            "resize_mode": resize_mode,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.local_weight < 0 or args.local_margin_threshold < 0:
        raise ValueError("Local reranking weight and threshold must be non-negative.")
    scales = parse_scales(args.scales)
    if args.embedding_db is not None and scales is not None:
        raise ValueError("--scales cannot change embeddings already stored in --embedding-db.")
    if args.save_embedding_db is not None and (
        args.checkpoint is None or len(args.checkpoint) != 1
    ):
        raise ValueError("--save-embedding-db requires exactly one --checkpoint.")

    rows, fieldnames = read_triplet_rows(args.triplets)
    if not rows:
        raise ValueError(f"No triplets found in {args.triplets}")
    if not fieldnames:
        raise ValueError(f"No triplet fields found in {args.triplets}")
    anchor_col, positive_col, negative_col = resolve_columns(
        fieldnames,
        args.anchor_col,
        args.positive_col,
        args.negative_col,
    )

    path_cache: dict[str, Path] = {}

    def cached_path(value: str) -> Path:
        if value not in path_cache:
            path_cache[value] = resolve_existing_image_path(args.image_root, value)
        return path_cache[value]

    unique_values: dict[str, Path] = {}
    for row in rows:
        for column in (anchor_col, positive_col, negative_col):
            unique_values.setdefault(row[column], cached_path(row[column]))

    value_embeddings: dict[str, torch.Tensor] | None = None
    if args.embedding_db is not None:
        value_embeddings = load_embedding_db(args.embedding_db)
        missing = [value for value in unique_values if value not in value_embeddings]
        if missing:
            raise KeyError(f"Embedding DB is missing {len(missing)} images. First missing: {missing[0]}")
    elif args.checkpoint is not None and len(args.checkpoint) == 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, image_size, mean, std, resize_mode = load_model(args.checkpoint[0], device)
        path_embeddings = embed_paths(
            paths=list(unique_values.values()),
            model=model,
            device=device,
            batch_size=args.batch_size,
            image_size=image_size,
            tta=args.tta,
            mean=mean,
            std=std,
            resize_mode=resize_mode,
            scales=scales,
        )
        value_embeddings = {
            value: path_embeddings[str(path)]
            for value, path in unique_values.items()
        }
        if args.save_embedding_db is not None:
            save_embedding_db(
                args.save_embedding_db,
                value_embeddings,
                args.checkpoint,
                args.tta,
                resize_mode,
                scales,
            )
        del path_embeddings
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    score_sums = [[0.0, 0.0] for _ in rows]
    if value_embeddings is not None:
        for index, row in enumerate(rows):
            anchor = value_embeddings[row[anchor_col]]
            positive = value_embeddings[row[positive_col]]
            negative = value_embeddings[row[negative_col]]
            score_sums[index][0] = float(torch.dot(anchor, positive))
            score_sums[index][1] = float(torch.dot(anchor, negative))
        score_count = 1
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert args.checkpoint is not None
        for checkpoint_path in args.checkpoint:
            model, image_size, mean, std, resize_mode = load_model(checkpoint_path, device)
            path_embeddings = embed_paths(
                paths=list(unique_values.values()),
                model=model,
                device=device,
                batch_size=args.batch_size,
                image_size=image_size,
                tta=args.tta,
                mean=mean,
                std=std,
                resize_mode=resize_mode,
                scales=scales,
            )
            for index, row in enumerate(rows):
                anchor = path_embeddings[str(unique_values[row[anchor_col]])]
                positive = path_embeddings[str(unique_values[row[positive_col]])]
                negative = path_embeddings[str(unique_values[row[negative_col]])]
                score_sums[index][0] += float(torch.dot(anchor, positive))
                score_sums[index][1] += float(torch.dot(anchor, negative))
            del path_embeddings
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        score_count = len(args.checkpoint)

    averaged_scores = [
        [scores[0] / score_count, scores[1] / score_count]
        for scores in score_sums
    ]
    if args.local_reranker == "lightglue":
        from triplet_landmark.local_matching import LightGlueGeometricReranker

        reranker = LightGlueGeometricReranker(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            features=args.local_features,
        )
        reranked = 0
        for index, (row, scores) in enumerate(zip(rows, averaged_scores)):
            if abs(scores[0] - scores[1]) > args.local_margin_threshold:
                continue
            anchor_path = unique_values[row[anchor_col]]
            positive_path = unique_values[row[positive_col]]
            negative_path = unique_values[row[negative_col]]
            scores[0] += args.local_weight * reranker.score(
                anchor_path, positive_path
            )
            scores[1] += args.local_weight * reranker.score(
                anchor_path, negative_path
            )
            reranked += 1
        print(f"LightGlue reranked ambiguous triplets: {reranked}/{len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    score_fields = {"sim_anchor_positive", "sim_anchor_negative"}
    output_fields = [field for field in fieldnames if field not in score_fields]
    output_fields += ["sim_anchor_positive", "sim_anchor_negative"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for row, scores in zip(rows, averaged_scores):
            row["sim_anchor_positive"] = f"{scores[0]:.8f}"
            row["sim_anchor_negative"] = f"{scores[1]:.8f}"
            writer.writerow(row)

    print(f"Saved triplet scores: {args.output}")


if __name__ == "__main__":
    main()
