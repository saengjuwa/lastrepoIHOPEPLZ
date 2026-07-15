from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triplet_landmark.train import make_checkpoint, resume_training_state


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 2)
        self.embedding_dim = 2


def training_args() -> argparse.Namespace:
    return argparse.Namespace(
        pooling="avg",
        gem_p=3.0,
        use_projection=False,
        embedding_dim=512,
        resize_mode="pad",
        epochs=5,
        batch_size=4,
        labels_per_batch=2,
        images_per_label=2,
        lr=1e-3,
        backbone_lr=1e-4,
        weight_decay=1e-4,
        triplet_weight=0.2,
        triplet_margin=0.2,
        hard_negative_ratio=0.7,
        hard_negative_weight=0.5,
        val_tta="flip",
        hard_val_fraction=0.5,
    )


class CheckpointResumeTests(unittest.TestCase):
    def test_optimizer_scheduler_and_epoch_are_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = training_args()
            labels = {"a": 0, "b": 1}
            model = TinyModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs
            )
            scaler = torch.amp.GradScaler("cuda", enabled=False)

            loss = model.backbone(torch.ones(1, 2)).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            payload = make_checkpoint(
                model,
                labels,
                args,
                epoch=2,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_accuracy=0.75,
                best_epoch=2,
            )
            checkpoint_path = Path(directory) / "resume.pt"
            torch.save(payload, checkpoint_path)

            restored_model = TinyModel()
            restored_optimizer = torch.optim.AdamW(
                restored_model.parameters(), lr=args.lr
            )
            restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                restored_optimizer, T_max=args.epochs
            )
            restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
            start_epoch, best_accuracy, best_epoch = resume_training_state(
                checkpoint_path,
                restored_model,
                restored_optimizer,
                restored_scheduler,
                restored_scaler,
                labels,
                args,
                torch.device("cpu"),
            )

            self.assertEqual(start_epoch, 3)
            self.assertEqual(best_accuracy, 0.75)
            self.assertEqual(best_epoch, 2)
            self.assertEqual(
                restored_scheduler.state_dict(), scheduler.state_dict()
            )
            for expected, actual in zip(
                model.parameters(), restored_model.parameters()
            ):
                self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
