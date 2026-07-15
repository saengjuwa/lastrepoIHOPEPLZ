from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triplet_landmark.data import (
    ImageRecord,
    class_disjoint_split,
    country_disjoint_split,
    load_split_manifest,
    make_inference_tensors,
    read_label_csv,
    resize_with_padding,
    save_split_manifest,
)


class DataSafetyTests(unittest.TestCase):
    def test_country_column_removes_korean_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "labels.csv"
            csv_path.write_text(
                "path,label,country_code\n"
                "a.jpg,1,FR\n"
                "b.jpg,1,FR\n"
                "k.jpg,2,KR\n",
                encoding="utf-8",
            )
            records, labels, _ = read_label_csv(csv_path, root)
            self.assertEqual([record.label for record in records], ["1", "1"])
            self.assertEqual(labels, {"1": 0})

    def test_missing_country_audit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "labels.csv"
            csv_path.write_text("path,label\na.jpg,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "korean-labels-file"):
                read_label_csv(csv_path, root)

    def test_blacklist_audits_csv_without_country_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "labels.csv"
            csv_path.write_text(
                "path,label\na.jpg,1\nk.jpg,2\n",
                encoding="utf-8",
            )
            records, _, _ = read_label_csv(
                csv_path, root, korean_label_ids={"2"}
            )
            self.assertEqual([record.label for record in records], ["1"])

    def test_duplicate_path_conflicting_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "labels.csv"
            csv_path.write_text(
                "path,label,country_code\na.jpg,1,FR\na.jpg,2,FR\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "conflicting labels"):
                read_label_csv(csv_path, root)

    def test_split_manifest_round_trip_and_csv_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "labels.csv"
            rows = ["path,label,country_code"]
            for label in range(4):
                rows.extend(
                    [
                        f"{label}_a.jpg,{label},FR",
                        f"{label}_b.jpg,{label},FR",
                    ]
                )
            csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            records, _, _ = read_label_csv(
                csv_path, root, min_images_per_label=2
            )
            train, val, _ = class_disjoint_split(records, 0.5, 42)
            manifest = root / "split.json"
            save_split_manifest(manifest, csv_path, train, val, 42, 0.5)
            loaded_train, loaded_val, _ = load_split_manifest(
                manifest, csv_path, records
            )
            self.assertEqual(
                {record.label for record in train},
                {record.label for record in loaded_train},
            )
            self.assertEqual(
                {record.label for record in val},
                {record.label for record in loaded_val},
            )
            csv_path.write_text(csv_path.read_text() + "x.jpg,9,FR\n")
            with self.assertRaisesRegex(ValueError, "different CSV"):
                load_split_manifest(manifest, csv_path, records)

    def test_padding_keeps_entire_image_and_flip_tta(self) -> None:
        image = Image.new("RGB", (8, 4), (255, 0, 0))
        padded = resize_with_padding(image, 8, (128, 128, 128))
        self.assertEqual(padded.size, (8, 8))
        self.assertEqual(padded.getpixel((0, 0)), (128, 128, 128))
        self.assertEqual(padded.getpixel((0, 3)), (255, 0, 0))
        tensors = make_inference_tensors(image, 8, tta="flip", resize_mode="pad")
        self.assertEqual(len(tensors), 2)
        with self.assertRaisesRegex(ValueError, "five_crop"):
            make_inference_tensors(image, 8, tta="five_crop", resize_mode="pad")
        with self.assertRaisesRegex(ValueError, "five_crop_flip"):
            make_inference_tensors(
                image, 8, tta="five_crop_flip", resize_mode="pad"
            )

    def test_five_crop_flip_tta_returns_ten_views(self) -> None:
        image = Image.new("RGB", (8, 4), (255, 0, 0))
        tensors = make_inference_tensors(
            image, 8, tta="five_crop_flip", resize_mode="center_crop"
        )
        self.assertEqual(len(tensors), 10)

    def test_country_disjoint_split_holds_out_whole_countries(self) -> None:
        records = [
            ImageRecord(Path("fr1.jpg"), "fr1", 0, "FR"),
            ImageRecord(Path("fr2.jpg"), "fr2", 1, "FR"),
            ImageRecord(Path("us1.jpg"), "us1", 2, "US"),
            ImageRecord(Path("us2.jpg"), "us2", 3, "US"),
        ]
        train, val, _ = country_disjoint_split(records, {"US"})
        self.assertEqual({record.country for record in train}, {"FR"})
        self.assertEqual({record.country for record in val}, {"US"})


if __name__ == "__main__":
    unittest.main()
