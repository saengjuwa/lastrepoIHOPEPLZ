from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triplet_landmark.predict_triplets import (
    embed_path_matrix,
    load_embedding_db,
    save_embedding_db,
)


class TinyEmbeddingModel(nn.Module):
    embedding_dim = 3

    def forward(self, images: torch.Tensor):
        embeddings = torch.nn.functional.normalize(images.mean(dim=(2, 3)), dim=1)
        return embeddings, torch.zeros((len(images), 1), device=images.device)


class EmbeddingDatabaseTests(unittest.TestCase):
    def test_image_paths_are_batched_into_normalized_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "red.jpg", root / "green.jpg"]
            Image.new("RGB", (12, 6), (255, 0, 0)).save(paths[0])
            Image.new("RGB", (6, 12), (0, 255, 0)).save(paths[1])
            matrix = embed_path_matrix(
                paths,
                TinyEmbeddingModel(),
                torch.device("cpu"),
                batch_size=2,
                image_size=8,
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                resize_mode="pad",
            )
            self.assertEqual(tuple(matrix.shape), (2, 3))
            self.assertTrue(
                torch.allclose(matrix.norm(dim=1), torch.ones(2), atol=1e-5)
            )

    def test_embedding_db_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.pt"
            values = {
                "a.jpg": torch.tensor([1.0, 0.0]),
                "b.jpg": torch.tensor([0.0, 1.0]),
            }
            save_embedding_db(
                path,
                values,
                [Path("model.pt")],
                "flip",
                "pad",
            )
            loaded = load_embedding_db(path)
            self.assertEqual(set(loaded), set(values))
            self.assertTrue(torch.equal(loaded["a.jpg"], values["a.jpg"]))

    def test_non_normalized_embedding_db_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            torch.save(
                {
                    "format_version": 1,
                    "values": ["a.jpg"],
                    "embeddings": torch.tensor([[2.0, 0.0]]),
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "non-normalized"):
                load_embedding_db(path)


if __name__ == "__main__":
    unittest.main()
