from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triplet_landmark.data import (
    ImageRecord,
    class_disjoint_split,
    read_label_csv,
    save_split_manifest,
)
from triplet_landmark import mine_hard_negatives
from triplet_landmark.train import (
    batch_hard_triplet_loss,
    make_validation_triplets,
    read_hard_negative_csv,
)
from triplet_landmark.mine_hard_negatives import different_label_neighbors


class TrainingSafetyTests(unittest.TestCase):
    def test_miner_uses_only_manifest_train_paths(self) -> None:
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
            all_records, _, _ = read_label_csv(csv_path, root)
            train_records, val_records, _ = class_disjoint_split(
                all_records, 0.5, 42
            )
            manifest = root / "split.json"
            save_split_manifest(
                manifest,
                csv_path,
                train_records,
                val_records,
                42,
                0.5,
            )
            output = root / "hard.csv"
            vectors = torch.tensor(
                [[1.0, 0.0], [0.9, 0.1], [0.2, 0.8], [0.0, 1.0]],
                dtype=torch.float32,
            )
            vectors = torch.nn.functional.normalize(vectors, dim=1)
            argv = [
                "mine_hard_negatives.py",
                "--checkpoint",
                str(root / "dummy.pt"),
                "--csv",
                str(csv_path),
                "--image-root",
                str(root),
                "--split-manifest",
                str(manifest),
                "--index-type",
                "flat",
                "--top-k",
                "1",
                "--output",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                mine_hard_negatives,
                "load_model",
                return_value=(object(), 300, (0.5,) * 3, (0.5,) * 3, "pad"),
            ), mock.patch.object(
                mine_hard_negatives,
                "embed_path_matrix",
                return_value=vectors,
            ):
                mine_hard_negatives.main()

            train_paths = {str(record.path) for record in train_records}
            with output.open(newline="", encoding="utf-8") as f:
                mined_rows = list(csv.DictReader(f))
            self.assertEqual(len(mined_rows), len(train_records))
            self.assertTrue(
                all(row["anchor_path"] in train_paths for row in mined_rows)
            )
            self.assertTrue(
                all(row["negative_path"] in train_paths for row in mined_rows)
            )

    def test_faiss_hnsw_search_filters_same_label(self) -> None:
        import faiss

        vectors = torch.tensor(
            [[1.0, 0.0], [0.99, 0.01], [0.8, 0.2], [-1.0, 0.0]],
            dtype=torch.float32,
        )
        vectors = torch.nn.functional.normalize(vectors, dim=1).numpy()
        index = faiss.IndexHNSWFlat(2, 8, faiss.METRIC_INNER_PRODUCT)
        index.add(vectors)
        similarities, indices = index.search(vectors[0:1], 4)
        neighbors = different_label_neighbors(
            similarities[0],
            indices[0],
            ["a", "a", "b", "c"],
            anchor_index=0,
            top_k=1,
        )
        self.assertEqual(neighbors[0][1], 2)

    def test_hard_negative_outside_train_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = root / "anchor.jpg"
            positive = root / "positive.jpg"
            leaked_validation = root / "validation.jpg"
            records = [
                ImageRecord(anchor, "1", 0),
                ImageRecord(positive, "1", 0),
            ]
            csv_path = root / "hard.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "anchor_path",
                        "anchor_label",
                        "negative_path",
                        "negative_label",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "anchor_path": anchor,
                        "anchor_label": "1",
                        "negative_path": leaked_validation,
                        "negative_label": "2",
                    }
                )
            with self.assertRaisesRegex(ValueError, "outside the fixed training split"):
                read_hard_negative_csv(csv_path, root, records)

    def test_fixed_hard_validation_uses_nearest_other_label(self) -> None:
        records = []
        path_to_label = {}
        for label in ("a", "b", "c"):
            for number in range(2):
                path = Path(f"{label}_{number}.jpg")
                records.append(ImageRecord(path, label, 0))
                path_to_label[path] = label
        centroids = {
            "a": torch.tensor([1.0, 0.0]),
            "b": torch.tensor([0.9, 0.1]),
            "c": torch.tensor([-1.0, 0.0]),
        }
        triplets = make_validation_triplets(
            records,
            max_triplets=3,
            seed=42,
            hard_negative_fraction=1.0,
            label_centroids=centroids,
        )
        negative_by_anchor_label = {
            path_to_label[anchor]: path_to_label[negative]
            for anchor, _, negative in triplets
        }
        self.assertEqual(negative_by_anchor_label["a"], "b")
        self.assertEqual(negative_by_anchor_label["b"], "a")

    def test_triplet_loss_rewards_closer_positive(self) -> None:
        good = torch.tensor(
            [[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0], [-0.99, 0.01]],
            dtype=torch.float32,
        )
        good = torch.nn.functional.normalize(good, dim=1)
        labels = torch.tensor([0, 0, 1, 1])
        self.assertEqual(batch_hard_triplet_loss(good, labels, 0.2).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
