from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triplet_landmark import train


class TinyLandmarkModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 8),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(8, num_classes)
        self.embedding_dim = 8

    def forward(self, images: torch.Tensor):
        features = self.backbone(images)
        return F.normalize(features, dim=1), self.classifier(features)


class TrainingMainSmokeTest(unittest.TestCase):
    def test_one_epoch_creates_audited_fixed_split_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "labels.csv"
            rows = ["path,label,country_code"]
            for label in range(4):
                for image_number in range(2):
                    image_path = root / f"{label}_{image_number}.jpg"
                    Image.new(
                        "RGB",
                        (12 + label, 8 + image_number),
                        (50 * label, 20 * image_number, 100),
                    ).save(image_path)
                    rows.append(f"{image_path.name},{label},FR")
            csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            output = root / "model.pt"
            split_manifest = root / "split.json"
            audit_output = root / "audit.json"
            argv = [
                "train.py",
                "--csv",
                str(csv_path),
                "--image-root",
                str(root),
                "--output",
                str(output),
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--labels-per-batch",
                "2",
                "--images-per-label",
                "2",
                "--num-workers",
                "0",
                "--select-best-triplet",
                "--val-fraction",
                "0.5",
                "--max-val-triplets",
                "4",
                "--hard-val-fraction",
                "0",
                "--val-tta",
                "none",
                "--resize-mode",
                "pad",
                "--split-manifest",
                str(split_manifest),
                "--data-audit-output",
                str(audit_output),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                train,
                "create_model",
                side_effect=lambda num_classes, **_: TinyLandmarkModel(num_classes),
            ):
                train.main()

            self.assertTrue(output.exists())
            self.assertTrue(split_manifest.exists())
            self.assertTrue(audit_output.exists())
            self.assertTrue((root / "model_last.pt").exists())
            checkpoint = torch.load(output, map_location="cpu")
            self.assertEqual(checkpoint["resize_mode"], "pad")
            self.assertIn("optimizer_state", checkpoint)
            self.assertIn("validation_metrics", checkpoint)


if __name__ == "__main__":
    unittest.main()
