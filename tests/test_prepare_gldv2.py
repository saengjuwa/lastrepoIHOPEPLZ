from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import tarfile
import types
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triplet_landmark import prepare_gldv2


def add_tar_image(archive: tarfile.TarFile, image_id: str, data: bytes) -> None:
    member = tarfile.TarInfo(
        name=f"{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"
    )
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))


class PrepareGldv2Tests(unittest.TestCase):
    def make_source(self, root: Path, md5_override: str | None = None) -> dict[str, str]:
        source = root / "official_source"
        source.mkdir()
        image_rows = [
            ("abc0000000000001", "https://example/a", "10"),
            ("abc0000000000002", "https://example/b", "10"),
            ("def0000000000001", "https://example/k1", "82"),
            ("def0000000000002", "https://example/k2", "82"),
            ("1230000000000001", "https://example/s", "99"),
        ]
        with (source / "train.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "url", "landmark_id"])
            writer.writerows(image_rows)

        archive_path = source / "images_000.tar"
        with tarfile.open(archive_path, "w") as archive:
            for image_id, _, _ in image_rows:
                add_tar_image(archive, image_id, f"bytes-{image_id}".encode())
            unsafe = tarfile.TarInfo("../../outside.txt")
            unsafe.size = 4
            archive.addfile(unsafe, io.BytesIO(b"nope"))

        checksum = hashlib.md5(archive_path.read_bytes()).hexdigest()
        (source / "md5.images_000.txt").write_text(
            f"{md5_override or checksum}  images_000.tar\n", encoding="utf-8"
        )
        base_url = source.as_uri()
        return {
            "metadata_url": f"{base_url}/train.csv",
            "archive_url_template": f"{base_url}/images_{{index:03d}}.tar",
            "md5_url_template": f"{base_url}/md5.images_{{index:03d}}.txt",
        }

    def test_end_to_end_filters_korean_and_small_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urls = self.make_source(root)
            korean_labels = root / "korean_label_ids.txt"
            korean_labels.write_text("# Korea\n82\n", encoding="utf-8")

            audit = prepare_gldv2.prepare_gldv2(
                dataset_root=root / "datasets",
                archive_count=1,
                korean_labels_file=korean_labels,
                **urls,
            )

            gldv2_root = root / "datasets" / "gldv2"
            manifest = gldv2_root / "train_labels.csv"
            with manifest.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(
                rows,
                [
                    {"path": "a/b/c/abc0000000000001.jpg", "label": "10"},
                    {"path": "a/b/c/abc0000000000002.jpg", "label": "10"},
                ],
            )
            for row in rows:
                self.assertTrue((gldv2_root / "train" / row["path"]).is_file())
            self.assertFalse((root / "outside.txt").exists())
            self.assertFalse((gldv2_root / "train" / "d").exists())
            self.assertEqual(audit["counts"]["korean_images_excluded"], 2)
            self.assertEqual(audit["counts"]["small_label_images_excluded"], 1)
            saved_audit = json.loads(
                (gldv2_root / "preparation_audit.json").read_text(encoding="utf-8")
            )
            self.assertTrue(saved_audit["assertions"]["korean_labels_excluded"])
            self.assertEqual(saved_audit["assertions"]["korean_label_overlap"], [])
            self.assertEqual(
                saved_audit["settings"]["korean_label_ids_count"], 1
            )
            self.assertEqual(
                saved_audit["settings"]["korean_label_ids_sha256"],
                hashlib.sha256(korean_labels.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                saved_audit["outputs"]["manifest_sha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )

    def test_generates_korean_labels_from_huggingface_places(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urls = self.make_source(root)
            calls: list[tuple[str, str]] = []
            def fake_load_dataset(name: str, split: str):
                calls.append((name, split))
                return [
                    {"id": 82, "country": "South Korea"},
                    {"id": 9001, "country": "North Korea"},
                    {"id": 10, "country": "France"},
                ]
            fake_datasets = types.SimpleNamespace(load_dataset=fake_load_dataset)
            with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
                audit = prepare_gldv2.prepare_gldv2(
                    dataset_root=root / "datasets",
                    archive_count=1,
                    korean_labels_file=None,
                    places_dataset="fake/places",
                    min_images_per_label=1,
                    **urls,
                )
            labels_file = root / "datasets" / "gldv2" / "korean_label_ids.txt"
            self.assertEqual(labels_file.read_text(encoding="utf-8").splitlines()[1:], ["82", "9001"])
            self.assertEqual(calls, [("fake/places", "train")])
            self.assertEqual(audit["settings"]["korean_labels_source"], "huggingface:fake/places")

    def test_rerun_reuses_verified_files_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urls = self.make_source(root)
            korean_labels = root / "korean_label_ids.txt"
            korean_labels.write_text("82\n", encoding="utf-8")
            arguments = {
                "dataset_root": root / "datasets",
                "archive_count": 1,
                "korean_labels_file": korean_labels,
                **urls,
            }
            prepare_gldv2.prepare_gldv2(**arguments)
            with mock.patch.object(
                prepare_gldv2.urllib.request,
                "urlopen",
                side_effect=AssertionError("network should not be used"),
            ):
                audit = prepare_gldv2.prepare_gldv2(**arguments)
            self.assertFalse(audit["archives"][0]["downloaded_this_run"])
            self.assertEqual(audit["counts"]["images_already_present"], 2)

    def test_md5_failure_never_creates_final_tar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urls = self.make_source(root, md5_override="0" * 32)
            korean_labels = root / "korean_label_ids.txt"
            korean_labels.write_text("82\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                prepare_gldv2.prepare_gldv2(
                    dataset_root=root / "datasets",
                    archive_count=1,
                    korean_labels_file=korean_labels,
                    **urls,
                )
            archive = root / "datasets" / "gldv2" / "archives" / "train" / "images_000.tar"
            self.assertFalse(archive.exists())
            self.assertTrue(archive.with_name(archive.name + ".part").exists())

    def test_incomplete_cached_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urls = self.make_source(root)
            metadata = root / "official_source" / "train.csv"
            with metadata.open(newline="", encoding="utf-8") as file:
                rows = list(csv.reader(file))
            with metadata.open("w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerows(rows[:-1])
            korean_labels = root / "korean_label_ids.txt"
            korean_labels.write_text("82\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "metadata is missing 1 image"):
                prepare_gldv2.prepare_gldv2(
                    dataset_root=root / "datasets",
                    archive_count=1,
                    korean_labels_file=korean_labels,
                    **urls,
                )
            self.assertFalse(
                (root / "datasets" / "gldv2" / "train_labels.csv").exists()
            )

    def test_archive_count_range_is_clear(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 500"):
            prepare_gldv2.validate_settings(0, 2)
        with self.assertRaisesRegex(ValueError, "between 1 and 500"):
            prepare_gldv2.validate_settings(501, 2)


if __name__ == "__main__":
    unittest.main()
