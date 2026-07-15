from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset, Sampler


MODEL_INPUT_SIZE = 300
MODEL_MEAN = (0.5, 0.5, 0.5)
MODEL_STD = (0.5, 0.5, 0.5)
RESIZE_MODES = {"center_crop", "pad"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str
    label_index: int
    country: str | None = None


@dataclass(frozen=True)
class DataAudit:
    audit_source: str
    total_rows_read: int
    kept_images: int
    kept_labels: int
    korean_rows_removed: int
    duplicate_rows_removed: int
    rows_removed_by_min_images: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "audit_source": self.audit_source,
            "total_rows_read": self.total_rows_read,
            "kept_images": self.kept_images,
            "kept_labels": self.kept_labels,
            "korean_rows_removed": self.korean_rows_removed,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "rows_removed_by_min_images": self.rows_removed_by_min_images,
        }


def _label_sort_key(label: str) -> tuple[int, int | str]:
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


def resolve_image_path(image_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or value.startswith(("/", "\\")):
        if path.exists():
            return path
        parts = path.parts
        for marker in ("train_10gb", "validation"):
            if marker not in parts:
                continue
            marker_index = parts.index(marker)
            candidates = [
                image_root.joinpath(*parts[marker_index:]),
                image_root.joinpath(*parts[marker_index + 1 :]),
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            if image_root.name == marker:
                return candidates[1]
            return candidates[0]
        return path
    return image_root / path


def canonical_path_key(path: Path) -> str:
    return str(path.resolve(strict=False)).casefold()


def read_korean_label_ids(path: Path) -> set[str]:
    labels = {
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not labels:
        raise ValueError(f"Korean label ID file is empty: {path}")
    return labels


def read_country_codes(path: Path) -> set[str]:
    countries = {
        line.strip().upper()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not countries:
        raise ValueError(f"Validation country file is empty: {path}")
    if "KR" in countries:
        raise ValueError("Korea cannot be used as a validation country.")
    return countries


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_label_csv(
    csv_path: Path,
    image_root: Path,
    min_images_per_label: int = 1,
    limit: int | None = None,
    country_column: str = "country_code",
    korean_label_ids: set[str] | None = None,
) -> tuple[list[ImageRecord], dict[str, int], DataAudit]:
    rows: list[tuple[Path, str, str | None]] = []
    labels_by_path: dict[str, str] = {}
    countries_by_label: dict[str, str] = {}
    korean_rows_removed = 0
    duplicate_rows_removed = 0
    total_rows_read = 0
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"path", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

        has_country_column = country_column in set(reader.fieldnames or [])
        if not has_country_column and korean_label_ids is None:
            raise ValueError(
                f"{csv_path} has no {country_column!r} column. "
                "Pass --korean-labels-file so Korean landmarks can be rejected."
            )

        for row in reader:
            total_rows_read += 1
            value = (row.get("path") or "").strip()
            label = (row.get("label") or "").strip()
            if not value or not label:
                raise ValueError(f"Empty path or label found in {csv_path}.")

            country = None
            if has_country_column:
                country = (row.get(country_column) or "").strip().upper()
                if not country:
                    raise ValueError(
                        f"Empty {country_column!r} for label {label!r} in {csv_path}."
                    )
                previous_country = countries_by_label.setdefault(label, country)
                if previous_country != country:
                    raise ValueError(
                        f"Label {label!r} has conflicting countries: "
                        f"{previous_country!r} and {country!r}."
                    )

            if country == "KR" or (
                korean_label_ids is not None and label in korean_label_ids
            ):
                korean_rows_removed += 1
                continue

            image_path = resolve_image_path(image_root, value)
            path_key = canonical_path_key(image_path)
            previous_label = labels_by_path.get(path_key)
            if previous_label is not None:
                if previous_label != label:
                    raise ValueError(
                        f"The same image path has conflicting labels: "
                        f"path={image_path} labels={previous_label!r},{label!r}"
                    )
                duplicate_rows_removed += 1
                continue

            labels_by_path[path_key] = label
            rows.append((image_path, label, country))
            if limit is not None and len(rows) >= limit:
                break

    counts = Counter(label for _, label, _ in rows)
    kept_labels = sorted(
        (label for label, count in counts.items() if count >= min_images_per_label),
        key=_label_sort_key,
    )
    label_to_index = {label: idx for idx, label in enumerate(kept_labels)}

    records = [
        ImageRecord(
            path=path,
            label=label,
            label_index=label_to_index[label],
            country=country,
        )
        for path, label, country in rows
        if label in label_to_index
    ]
    if not records:
        raise ValueError("No training records left after filtering labels.")
    audit_source = country_column if has_country_column else "korean-labels-file"
    audit = DataAudit(
        audit_source=audit_source,
        total_rows_read=total_rows_read,
        kept_images=len(records),
        kept_labels=len(label_to_index),
        korean_rows_removed=korean_rows_removed,
        duplicate_rows_removed=duplicate_rows_removed,
        rows_removed_by_min_images=len(rows) - len(records),
    )
    print(
        f"data_audit source={audit_source} kept_images={len(records)} "
        f"kept_labels={len(label_to_index)} korean_rows_removed={korean_rows_removed} "
        f"duplicate_rows_removed={duplicate_rows_removed}",
        flush=True,
    )
    return records, label_to_index, audit


def class_disjoint_split(
    records: list[ImageRecord],
    val_fraction: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord], dict[str, int]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")

    labels = sorted({record.label for record in records}, key=_label_sort_key)
    if len(labels) < 3:
        raise ValueError("Best-triplet selection requires at least three labels.")

    random.Random(seed).shuffle(labels)
    val_count = min(max(2, round(len(labels) * val_fraction)), len(labels) - 1)
    val_labels = set(labels[:val_count])
    train_labels = sorted(set(labels[val_count:]), key=_label_sort_key)
    label_to_index = {label: index for index, label in enumerate(train_labels)}

    train_records = [
        ImageRecord(
            record.path,
            record.label,
            label_to_index[record.label],
            record.country,
        )
        for record in records
        if record.label in label_to_index
    ]
    val_records = [record for record in records if record.label in val_labels]
    return train_records, val_records, label_to_index


def country_disjoint_split(
    records: list[ImageRecord],
    val_countries: set[str],
) -> tuple[list[ImageRecord], list[ImageRecord], dict[str, int]]:
    if not val_countries or "KR" in val_countries:
        raise ValueError("Validation countries must be non-empty and must exclude KR.")
    if any(record.country is None for record in records):
        raise ValueError("Country-disjoint validation requires a country column.")
    val_labels = {
        record.label for record in records if record.country in val_countries
    }
    train_labels = {record.label for record in records} - val_labels
    if len(train_labels) < 2 or len(val_labels) < 2:
        raise ValueError(
            "Country-disjoint validation needs at least two train and two validation labels."
        )
    return split_records_by_labels(records, train_labels, val_labels)


def split_records_by_labels(
    records: list[ImageRecord],
    train_labels: set[str],
    val_labels: set[str],
) -> tuple[list[ImageRecord], list[ImageRecord], dict[str, int]]:
    if train_labels & val_labels:
        raise ValueError("Split manifest has labels in both train and validation sets.")
    record_labels = {record.label for record in records}
    manifest_labels = train_labels | val_labels
    if record_labels != manifest_labels:
        missing = sorted(record_labels - manifest_labels, key=_label_sort_key)
        extra = sorted(manifest_labels - record_labels, key=_label_sort_key)
        raise ValueError(
            "Split manifest does not match the filtered dataset. "
            f"missing_labels={missing[:5]} extra_labels={extra[:5]}"
        )

    ordered_train_labels = sorted(train_labels, key=_label_sort_key)
    label_to_index = {
        label: index for index, label in enumerate(ordered_train_labels)
    }
    train_records = [
        ImageRecord(
            record.path,
            record.label,
            label_to_index[record.label],
            record.country,
        )
        for record in records
        if record.label in train_labels
    ]
    val_records = [record for record in records if record.label in val_labels]
    return train_records, val_records, label_to_index


def save_split_manifest(
    path: Path,
    csv_path: Path,
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    seed: int,
    val_fraction: float,
    split_strategy: str = "class_random",
    val_countries: set[str] | None = None,
) -> None:
    train_labels = sorted({record.label for record in train_records}, key=_label_sort_key)
    val_labels = sorted({record.label for record in val_records}, key=_label_sort_key)
    if set(train_labels) & set(val_labels):
        raise ValueError("Cannot save a split with overlapping labels.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "csv_sha256": file_sha256(csv_path),
                "seed": seed,
                "val_fraction": val_fraction,
                "split_strategy": split_strategy,
                "val_countries": sorted(val_countries or []),
                "train_labels": train_labels,
                "val_labels": val_labels,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_split_manifest(
    path: Path,
    csv_path: Path,
    records: list[ImageRecord],
) -> tuple[list[ImageRecord], list[ImageRecord], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported split manifest format: {path}")
    expected_hash = file_sha256(csv_path)
    if payload.get("csv_sha256") != expected_hash:
        raise ValueError(
            f"Split manifest was created for a different CSV: {path}"
        )
    train_labels = {str(label) for label in payload.get("train_labels", [])}
    val_labels = {str(label) for label in payload.get("val_labels", [])}
    if not train_labels or len(val_labels) < 2:
        raise ValueError("Split manifest needs train labels and at least two validation labels.")
    return split_records_by_labels(records, train_labels, val_labels)


def resize_with_padding(
    image: Image.Image,
    image_size: int,
    fill: tuple[int, int, int],
) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size), fill)
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


class ImageTransform:
    def __init__(
        self,
        image_size: int,
        train: bool,
        mean: tuple[float, float, float] = MODEL_MEAN,
        std: tuple[float, float, float] = MODEL_STD,
        resize_mode: str = "center_crop",
        augmentation: str = "basic",
    ) -> None:
        if resize_mode not in RESIZE_MODES:
            raise ValueError(f"Unsupported resize mode: {resize_mode}")
        if augmentation not in {"basic", "weak"}:
            raise ValueError("augmentation must be 'basic' or 'weak'.")
        self.image_size = image_size
        self.train = train
        self.resize_mode = resize_mode
        self.augmentation = augmentation
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.fill = tuple(round(value * 255) for value in mean)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.resize_mode == "pad":
            if self.train and random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = resize_with_padding(image, self.image_size, self.fill)
        else:
            image = self._crop(image)
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )

        if self.train and self.augmentation == "weak":
            image = self._weak_pil_augmentation(image)
        tensor = image_to_normalized_tensor(image, self.image_size, self.mean, self.std)
        if self.train and self.augmentation == "weak" and random.random() < 0.2:
            self._random_erasing(tensor)
        return tensor

    def _weak_pil_augmentation(self, image: Image.Image) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.1:
            image = ImageOps.grayscale(image).convert("RGB")
        if random.random() < 0.3:
            image = image.rotate(
                random.uniform(-5.0, 5.0),
                resample=Image.Resampling.BICUBIC,
                fillcolor=self.fill,
            )
        return image

    def _random_erasing(self, tensor: torch.Tensor) -> None:
        area = self.image_size * self.image_size
        erase_area = random.uniform(0.02, 0.1) * area
        aspect = random.uniform(0.5, 2.0)
        height = min(self.image_size, max(1, round((erase_area * aspect) ** 0.5)))
        width = min(self.image_size, max(1, round((erase_area / aspect) ** 0.5)))
        top = random.randint(0, self.image_size - height)
        left = random.randint(0, self.image_size - width)
        tensor[:, top : top + height, left : left + width] = 0.0

    def _crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = min(width, height)
        if self.train:
            side = max(1, int(side * random.uniform(0.75, 1.0)))
            left = random.randint(0, max(0, width - side))
            top = random.randint(0, max(0, height - side))
            image = image.crop((left, top, left + side, top + side))
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return image

        left = (width - side) // 2
        top = (height - side) // 2
        return image.crop((left, top, left + side, top + side))


def image_to_normalized_tensor(
    image: Image.Image,
    image_size: int,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    if mean is None:
        mean = torch.tensor(MODEL_MEAN).view(3, 1, 1)
    if std is None:
        std = torch.tensor(MODEL_STD).view(3, 1, 1)
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = tensor.view(image_size, image_size, 3).permute(2, 0, 1)
    tensor = tensor.float().div(255.0)
    return (tensor - mean) / std


def make_inference_tensors(
    image: Image.Image,
    image_size: int,
    tta: str = "none",
    mean: tuple[float, float, float] = MODEL_MEAN,
    std: tuple[float, float, float] = MODEL_STD,
    resize_mode: str = "center_crop",
) -> list[torch.Tensor]:
    if resize_mode not in RESIZE_MODES:
        raise ValueError(f"Unsupported resize mode: {resize_mode}")
    image = image.convert("RGB")
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    if resize_mode == "pad":
        if tta in {"five_crop", "five_crop_flip"}:
            raise ValueError(
                "five_crop and five_crop_flip TTA are only available with center_crop."
            )
        if tta not in {"none", "flip"}:
            raise ValueError(f"Unsupported TTA mode: {tta}")
        fill = tuple(round(value * 255) for value in mean)
        resized = resize_with_padding(image, image_size, fill)
        tensors = [
            image_to_normalized_tensor(resized, image_size, mean_tensor, std_tensor)
        ]
        if tta == "flip":
            tensors.append(
                image_to_normalized_tensor(
                    resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                    image_size,
                    mean_tensor,
                    std_tensor,
                )
            )
        return tensors

    width, height = image.size
    side = min(width, height)
    center_left = (width - side) // 2
    center_top = (height - side) // 2
    boxes = [(center_left, center_top, center_left + side, center_top + side)]

    if tta in {"five_crop", "five_crop_flip"}:
        boxes = [
            (0, 0, side, side),
            (width - side, 0, width, side),
            (0, height - side, side, height),
            (width - side, height - side, width, height),
            (center_left, center_top, center_left + side, center_top + side),
        ]
    elif tta not in {"none", "flip"}:
        raise ValueError(f"Unsupported TTA mode: {tta}")

    tensors: list[torch.Tensor] = []
    for box in boxes:
        crop = image.crop(box).resize((image_size, image_size), Image.Resampling.BICUBIC)
        tensors.append(
            image_to_normalized_tensor(crop, image_size, mean_tensor, std_tensor)
        )
        if tta in {"flip", "five_crop_flip"}:
            flipped = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            tensors.append(
                image_to_normalized_tensor(flipped, image_size, mean_tensor, std_tensor)
            )
    return tensors


class LandmarkDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform: ImageTransform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image)
        return tensor, torch.tensor(record.label_index, dtype=torch.long)


class BalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        records: list[ImageRecord],
        labels_per_batch: int,
        images_per_label: int,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        if labels_per_batch <= 0 or images_per_label <= 0:
            raise ValueError("labels_per_batch and images_per_label must be positive.")

        self.labels_per_batch = labels_per_batch
        self.images_per_label = images_per_label
        self.seed = seed
        self.epoch = 0
        self.indices_by_label: dict[int, list[int]] = {}
        for index, record in enumerate(records):
            self.indices_by_label.setdefault(record.label_index, []).append(index)
        if not self.indices_by_label:
            raise ValueError("BalancedBatchSampler received no records.")

        batch_size = labels_per_batch * images_per_label
        self.batches_per_epoch = batches_per_epoch or max(1, len(records) // batch_size)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        labels = list(self.indices_by_label)
        for _ in range(self.batches_per_epoch):
            if len(labels) >= self.labels_per_batch:
                batch_labels = rng.sample(labels, self.labels_per_batch)
            else:
                batch_labels = labels.copy()
                batch_labels.extend(
                    rng.choice(labels)
                    for _ in range(self.labels_per_batch - len(batch_labels))
                )
                rng.shuffle(batch_labels)

            batch: list[int] = []
            for label in batch_labels:
                indices = self.indices_by_label[label]
                if len(indices) >= self.images_per_label:
                    batch.extend(rng.sample(indices, self.images_per_label))
                else:
                    batch.extend(rng.choice(indices) for _ in range(self.images_per_label))
            yield batch
