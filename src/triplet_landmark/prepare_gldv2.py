from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable

GLDV2_ARCHIVE_COUNT = 500
DEFAULT_PLACES_DATASET = "visheratin/google_landmarks_places"
TRAIN_METADATA_URL = "https://s3.amazonaws.com/google-landmark/metadata/train.csv"
TRAIN_ARCHIVE_URL = (
    "https://s3.amazonaws.com/google-landmark/train/images_{index:03d}.tar"
)
TRAIN_MD5_URL = (
    "https://s3.amazonaws.com/google-landmark/md5sum/train/"
    "md5.images_{index:03d}.txt"
)
_MD5_PATTERN = re.compile(r"\b([0-9a-fA-F]{32})\b")
_IMAGE_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{3,}$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the first N official GLDv2 training TAR files, remove "
            "Korean landmark labels, and create a path,label training CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets"),
        help=(
            "Parent dataset folder. Files are saved below DATASET_ROOT/gldv2; "
            "for example D:/datasets creates D:/datasets/gldv2."
        ),
    )
    parser.add_argument(
        "--archive-count",
        type=int,
        required=True,
        help=(
            "Number of training shards to download, starting at images_000.tar. "
            "Valid range: 1-500. Each official shard is approximately 1 GB."
        ),
    )
    parser.add_argument(
        "--korean-labels-file",
        type=Path,
        default=None,
        help=(
            "Path for the generated Korean landmark_id file. If omitted, "
            "DATASET_ROOT/gldv2/korean_label_ids.txt is used."
        ),
    )
    parser.add_argument(
        "--places-dataset",
        default=DEFAULT_PLACES_DATASET,
        help="HuggingFace dataset containing country and landmark id columns.",
    )
    parser.add_argument("--refresh-korean-labels", action="store_true")
    parser.add_argument(
        "--min-images-per-label",
        type=int,
        default=2,
        help="Discard labels with fewer selected images than this value.",
    )
    return parser


def validate_settings(archive_count: int, min_images_per_label: int) -> None:
    if not 1 <= archive_count <= GLDV2_ARCHIVE_COUNT:
        raise ValueError(
            f"archive_count must be between 1 and {GLDV2_ARCHIVE_COUNT}; "
            f"got {archive_count}."
        )
    if min_images_per_label < 1:
        raise ValueError(
            "min_images_per_label must be at least 1; "
            f"got {min_images_per_label}."
        )


def read_korean_label_ids(path: Path) -> set[str]:
    labels = {
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not labels:
        raise ValueError(f"Korean label ID file is empty: {path}")
    return labels


def _normalize_places_id(value: object) -> str:
    if value is None:
        raise ValueError("HuggingFace places row has an empty id.")
    raw = str(value).strip()
    if not raw:
        raise ValueError("HuggingFace places row has an empty id.")
    try:
        return str(int(raw))
    except ValueError:
        return raw


def load_korean_labels_from_places(dataset_name: str) -> set[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Automatic Korean filtering needs HuggingFace datasets. "
            "Install it with: py -3 -m pip install -r requirements.txt"
        ) from exc
    dataset = load_dataset(dataset_name, split="train")
    korean_countries = {"South Korea", "North Korea"}
    korean_ids: set[str] = set()
    for row in dataset:
        if "country" not in row or "id" not in row:
            raise ValueError(
                f"HuggingFace dataset {dataset_name!r} must contain "
                "country and id columns."
            )
        country = str(row["country"] or "").strip()
        if country in korean_countries:
            korean_ids.add(_normalize_places_id(row["id"]))
    if not korean_ids:
        raise ValueError(
            f"No South Korea/North Korea landmark ids found in {dataset_name!r}."
        )
    return korean_ids


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _part_path(path)
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def ensure_korean_label_file(
    path: Path,
    dataset_name: str,
    refresh: bool,
) -> tuple[Path, str]:
    path = path.expanduser().resolve()
    if path.is_file() and not refresh:
        read_korean_label_ids(path)
        return path, "existing-file"
    korean_ids = load_korean_labels_from_places(dataset_name)
    contents = "# Auto-generated from " + dataset_name + "\n"
    contents += "\n".join(sorted(korean_ids)) + "\n"
    write_text_atomic(path, contents)
    return path, "huggingface:" + dataset_name


def _part_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".part")


def download_file_atomic(url: str, destination: Path) -> bool:
    """Download a file without ever exposing an incomplete final file.

    Returns True when a download occurred and False when the final file already
    existed. A failed download leaves only the .part file, which is overwritten
    on the next run.
    """

    if destination.is_file():
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = _part_path(destination)
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(destination)
    except Exception:
        print(f"Download stopped. Partial file kept at: {partial}")
        raise
    return True


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_expected_md5(path: Path) -> str:
    match = _MD5_PATTERN.search(path.read_text(encoding="utf-8-sig"))
    if match is None:
        raise ValueError(f"No MD5 value was found in {path}.")
    return match.group(1).lower()


def ensure_verified_archive(
    index: int,
    archive_directory: Path,
    checksum_directory: Path,
    archive_url_template: str = TRAIN_ARCHIVE_URL,
    md5_url_template: str = TRAIN_MD5_URL,
) -> tuple[Path, str, bool]:
    archive_name = f"images_{index:03d}.tar"
    archive_path = archive_directory / archive_name
    checksum_path = checksum_directory / f"md5.images_{index:03d}.txt"

    download_file_atomic(md5_url_template.format(index=index), checksum_path)
    expected_md5 = read_expected_md5(checksum_path)

    if archive_path.is_file() and file_md5(archive_path) == expected_md5:
        print(f"Verified archive already exists: {archive_path}")
        return archive_path, expected_md5, False

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial = _part_path(archive_path)
    print(f"Downloading {archive_url_template.format(index=index)}")
    try:
        with urllib.request.urlopen(
            archive_url_template.format(index=index)
        ) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except Exception:
        print(f"Download stopped. Partial file kept at: {partial}")
        raise

    actual_md5 = file_md5(partial)
    if actual_md5 != expected_md5:
        raise ValueError(
            f"MD5 mismatch for {archive_name}: expected {expected_md5}, "
            f"downloaded {actual_md5}. The final TAR was not replaced; "
            f"rerun the command to overwrite {partial}."
        )
    partial.replace(archive_path)
    return archive_path, expected_md5, True


def image_id_from_tar_member(member: tarfile.TarInfo) -> str | None:
    if not member.isfile():
        return None
    filename = PurePosixPath(member.name).name
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg"}:
        return None
    image_id = PurePosixPath(filename).stem
    if _IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        return None
    return image_id.lower()


def image_relative_path(image_id: str) -> Path:
    if _IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise ValueError(f"Invalid GLDv2 image ID: {image_id!r}")
    image_id = image_id.lower()
    return Path(image_id[0]) / image_id[1] / image_id[2] / f"{image_id}.jpg"


def collect_archive_image_ids(archive_paths: Iterable[Path]) -> set[str]:
    image_ids: set[str] = set()
    for archive_path in archive_paths:
        print(f"Reading image IDs: {archive_path.name}")
        with tarfile.open(archive_path, mode="r") as archive:
            for member in archive:
                image_id = image_id_from_tar_member(member)
                if image_id is not None:
                    image_ids.add(image_id)
    return image_ids


def read_selected_metadata(
    metadata_path: Path,
    selected_image_ids: set[str],
    korean_label_ids: set[str],
) -> tuple[dict[str, str], int, int, int]:
    labels_by_image: dict[str, str] = {}
    seen_labels_by_image: dict[str, str] = {}
    korean_images_excluded = 0
    duplicate_rows = 0
    with metadata_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"id", "url", "landmark_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Official metadata CSV is missing columns {sorted(missing)}: "
                f"{metadata_path}"
            )
        for row in reader:
            image_id = (row.get("id") or "").strip().lower()
            if image_id not in selected_image_ids:
                continue
            label = (row.get("landmark_id") or "").strip()
            if not label:
                raise ValueError(
                    f"Empty landmark_id for selected image {image_id} in "
                    f"{metadata_path}."
                )
            previous = seen_labels_by_image.get(image_id)
            if previous is not None:
                if previous != label:
                    raise ValueError(
                        f"Image {image_id} has conflicting landmark_id values: "
                        f"{previous!r} and {label!r}."
                    )
                duplicate_rows += 1
                continue
            seen_labels_by_image[image_id] = label
            if label in korean_label_ids:
                korean_images_excluded += 1
                continue
            labels_by_image[image_id] = label
    return (
        labels_by_image,
        korean_images_excluded,
        duplicate_rows,
        len(seen_labels_by_image),
    )


def extract_selected_images(
    archive_paths: Iterable[Path],
    image_root: Path,
    allowed_image_ids: set[str],
) -> tuple[int, int]:
    extracted = 0
    already_present = 0
    remaining = set(allowed_image_ids)
    for archive_path in archive_paths:
        print(f"Extracting selected images: {archive_path.name}")
        with tarfile.open(archive_path, mode="r") as archive:
            for member in archive:
                image_id = image_id_from_tar_member(member)
                if image_id is None or image_id not in remaining:
                    continue
                destination = image_root / image_relative_path(image_id)
                if destination.is_file():
                    already_present += 1
                    remaining.remove(image_id)
                    continue
                source = archive.extractfile(member)
                if source is None:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = _part_path(destination)
                with source, partial.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                partial.replace(destination)
                remaining.remove(image_id)
                extracted += 1

    if remaining:
        sample = ", ".join(sorted(remaining)[:5])
        raise RuntimeError(
            f"{len(remaining)} selected images were not found during extraction. "
            f"Examples: {sample}"
        )
    return extracted, already_present


def write_manifest(
    manifest_path: Path,
    image_ids: Iterable[str],
    labels_by_image: dict[str, str],
) -> int:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    partial = _part_path(manifest_path)
    count = 0
    with partial.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label"])
        writer.writeheader()
        for image_id in sorted(image_ids):
            writer.writerow(
                {
                    "path": image_relative_path(image_id).as_posix(),
                    "label": labels_by_image[image_id],
                }
            )
            count += 1
    partial.replace(manifest_path)
    return count


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _part_path(path)
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def prepare_gldv2(
    dataset_root: Path,
    archive_count: int,
    korean_labels_file: Path | None = None,
    min_images_per_label: int = 2,
    places_dataset: str = DEFAULT_PLACES_DATASET,
    refresh_korean_labels: bool = False,
    metadata_url: str = TRAIN_METADATA_URL,
    archive_url_template: str = TRAIN_ARCHIVE_URL,
    md5_url_template: str = TRAIN_MD5_URL,
) -> dict[str, object]:
    validate_settings(archive_count, min_images_per_label)
    gldv2_root = dataset_root.expanduser().resolve() / "gldv2"
    requested_labels_file = korean_labels_file or (
        gldv2_root / "korean_label_ids.txt"
    )
    korean_labels_file, korean_labels_source = ensure_korean_label_file(
        requested_labels_file, places_dataset, refresh_korean_labels
    )
    korean_label_ids = read_korean_label_ids(korean_labels_file)
    metadata_path = gldv2_root / "metadata" / "train.csv"
    archive_directory = gldv2_root / "archives" / "train"
    checksum_directory = gldv2_root / "metadata" / "md5" / "train"
    image_root = gldv2_root / "train"
    manifest_path = gldv2_root / "train_labels.csv"
    audit_path = gldv2_root / "preparation_audit.json"

    download_file_atomic(metadata_url, metadata_path)
    archive_paths: list[Path] = []
    archive_audit: list[dict[str, object]] = []
    for index in range(archive_count):
        archive_path, expected_md5, downloaded = ensure_verified_archive(
            index,
            archive_directory,
            checksum_directory,
            archive_url_template,
            md5_url_template,
        )
        archive_paths.append(archive_path)
        archive_audit.append(
            {
                "index": index,
                "file": str(archive_path),
                "md5": expected_md5,
                "downloaded_this_run": downloaded,
            }
        )

    selected_ids = collect_archive_image_ids(archive_paths)
    (
        labels_by_image,
        korean_images_excluded,
        duplicate_metadata_rows,
        metadata_matches,
    ) = read_selected_metadata(metadata_path, selected_ids, korean_label_ids)
    metadata_missing = len(selected_ids) - metadata_matches
    if metadata_missing:
        raise RuntimeError(
            f"Official metadata is missing {metadata_missing} image ID(s) "
            "found in the selected TAR files. The cached train.csv may be "
            f"incomplete; delete {metadata_path} and run this command again."
        )
    label_counts = Counter(labels_by_image.values())
    kept_labels = {
        label
        for label, count in label_counts.items()
        if count >= min_images_per_label
    }
    korean_overlap = sorted(kept_labels & korean_label_ids)
    if korean_overlap:
        raise AssertionError(
            "Korean landmark labels remained after filtering: "
            + ", ".join(korean_overlap[:10])
        )
    allowed_ids = {
        image_id
        for image_id, label in labels_by_image.items()
        if label in kept_labels
    }
    extracted, already_present = extract_selected_images(
        archive_paths, image_root, allowed_ids
    )
    manifest_rows = write_manifest(manifest_path, allowed_ids, labels_by_image)

    audit: dict[str, object] = {
        "format_version": 1,
        "source": {
            "metadata_url": metadata_url,
            "archive_url_template": archive_url_template,
            "md5_url_template": md5_url_template,
        },
        "settings": {
            "archive_count": archive_count,
            "archive_indexes": list(range(archive_count)),
            "min_images_per_label": min_images_per_label,
            "korean_labels_file": str(korean_labels_file),
            "korean_labels_source": korean_labels_source,
            "places_dataset": places_dataset,
            "korean_label_ids_count": len(korean_label_ids),
            "korean_label_ids_sha256": file_sha256(korean_labels_file),
        },
        "outputs": {
            "image_root": str(image_root),
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
        },
        "counts": {
            "tar_image_ids": len(selected_ids),
            "metadata_matches": metadata_matches,
            "metadata_missing_for_tar_ids": metadata_missing,
            "duplicate_metadata_rows": duplicate_metadata_rows,
            "korean_images_excluded": korean_images_excluded,
            "small_label_images_excluded": len(labels_by_image) - len(allowed_ids),
            "kept_labels": len(kept_labels),
            "kept_images": manifest_rows,
            "images_extracted_this_run": extracted,
            "images_already_present": already_present,
        },
        "assertions": {
            "korean_label_overlap": korean_overlap,
            "korean_labels_excluded": not korean_overlap,
        },
        "archives": archive_audit,
    }
    write_json_atomic(audit_path, audit)
    print(f"Ready: {manifest_rows} images across {len(kept_labels)} labels")
    print(f"Image root: {image_root}")
    print(f"Training CSV: {manifest_path}")
    print(f"Audit JSON: {audit_path}")
    return audit


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_gldv2(
        dataset_root=args.dataset_root,
        archive_count=args.archive_count,
        korean_labels_file=args.korean_labels_file,
        min_images_per_label=args.min_images_per_label,
        places_dataset=args.places_dataset,
        refresh_korean_labels=args.refresh_korean_labels,
    )


if __name__ == "__main__":
    main()


