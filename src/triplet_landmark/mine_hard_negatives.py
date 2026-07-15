from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from triplet_landmark.data import (
    file_sha256,
    load_split_manifest,
    read_korean_label_ids,
    read_label_csv,
)
from triplet_landmark.predict_triplets import embed_path_matrix, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine hard negatives from the fixed GLDv2 training split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--country-col", default="country_code")
    parser.add_argument("--korean-labels-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/hard_negatives.csv"))
    parser.add_argument("--save-index", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--index-type", choices=["hnsw", "flat"], default="hnsw")
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--search-multiplier", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def different_label_neighbors(
    similarities,
    neighbor_indices,
    labels: list[str],
    anchor_index: int,
    top_k: int,
) -> list[tuple[float, int]]:
    selected: list[tuple[float, int]] = []
    seen: set[int] = set()
    anchor_label = labels[anchor_index]
    for similarity, neighbor_index in zip(similarities, neighbor_indices):
        neighbor_index = int(neighbor_index)
        if neighbor_index < 0 or neighbor_index in seen:
            continue
        seen.add(neighbor_index)
        if labels[neighbor_index] == anchor_label:
            continue
        selected.append((float(similarity), neighbor_index))
        if len(selected) == top_k:
            break
    return selected


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.hnsw_m <= 0 or args.search_multiplier <= 0:
        raise ValueError("--hnsw-m and --search-multiplier must be positive.")

    try:
        import faiss
    except ImportError as exc:
        raise ImportError("Install faiss-cpu with: pip install -r requirements.txt") from exc

    korean_label_ids = (
        read_korean_label_ids(args.korean_labels_file)
        if args.korean_labels_file is not None
        else None
    )
    all_records, _, _ = read_label_csv(
        csv_path=args.csv,
        image_root=args.image_root,
        min_images_per_label=2,
        limit=args.limit,
        country_column=args.country_col,
        korean_label_ids=korean_label_ids,
    )
    records, _, _ = load_split_manifest(
        args.split_manifest,
        args.csv,
        all_records,
    )
    paths = [record.path for record in records]
    labels = [record.label for record in records]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size, mean, std, resize_mode = load_model(args.checkpoint, device)
    embeddings = embed_path_matrix(
        paths=paths,
        model=model,
        device=device,
        batch_size=args.batch_size,
        image_size=image_size,
        mean=mean,
        std=std,
        resize_mode=resize_mode,
    ).contiguous()
    vectors = embeddings.numpy()

    dimension = int(vectors.shape[1])
    if args.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(
            dimension,
            args.hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.hnsw.efConstruction = 200
    else:
        index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    initial_search_k = min(
        len(records),
        max(args.top_k * args.search_multiplier + 1, 128),
    )
    if args.index_type == "hnsw":
        index.hnsw.efSearch = max(64, initial_search_k)
    similarities, neighbor_indices = index.search(vectors, initial_search_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    short_anchors = 0
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "anchor_path",
                "anchor_label",
                "negative_path",
                "negative_label",
                "similarity",
            ],
        )
        writer.writeheader()

        for anchor_index, record in enumerate(records):
            neighbors = different_label_neighbors(
                similarities[anchor_index],
                neighbor_indices[anchor_index],
                labels,
                anchor_index,
                args.top_k,
            )
            search_k = initial_search_k
            while len(neighbors) < args.top_k and search_k < len(records):
                search_k = min(len(records), search_k * 2)
                if args.index_type == "hnsw":
                    index.hnsw.efSearch = max(index.hnsw.efSearch, search_k)
                retry_similarities, retry_indices = index.search(
                    vectors[anchor_index : anchor_index + 1],
                    search_k,
                )
                neighbors = different_label_neighbors(
                    retry_similarities[0],
                    retry_indices[0],
                    labels,
                    anchor_index,
                    args.top_k,
                )
            if len(neighbors) < args.top_k:
                short_anchors += 1

            for similarity, negative_index in neighbors:
                writer.writerow(
                    {
                        "anchor_path": str(paths[anchor_index]),
                        "anchor_label": record.label,
                        "negative_path": str(paths[negative_index]),
                        "negative_label": records[negative_index].label,
                        "similarity": f"{similarity:.8f}",
                    }
                )
                written += 1

    if args.save_index is not None:
        args.save_index.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(args.save_index))

    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "source_csv_sha256": file_sha256(args.csv),
                "split_manifest": str(args.split_manifest),
                "checkpoint": str(args.checkpoint),
                "index_type": args.index_type,
                "anchors": len(records),
                "top_k": args.top_k,
                "rows": written,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved hard negatives: {args.output}")
    print(
        f"anchors={len(records)} rows={written} top_k={args.top_k} "
        f"short_anchors={short_anchors} index_type={args.index_type}"
    )


if __name__ == "__main__":
    main()
