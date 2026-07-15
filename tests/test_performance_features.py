from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triplet_landmark.data import ImageRecord
from triplet_landmark.model import (
    ArcMarginClassifier,
    DOLGAggregator,
    SALADAggregator,
    create_model,
)
from triplet_landmark.predict_triplets import parse_scales
from triplet_landmark.train import (
    CrossBatchMemory,
    ProxyAnchorLoss,
    compute_validation_label_centroids,
    cross_batch_memory_loss,
    load_initial_weights,
    supervised_contrastive_loss,
)


class FakeDinoBackbone(torch.nn.Module):
    num_features = 8
    num_prefix_tokens = 1

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)]
        )
        self.norm = torch.nn.LayerNorm(8)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.size(0)
        values = torch.arange(21 * self.num_features, dtype=images.dtype)
        return values.reshape(1, 21, self.num_features).repeat(batch, 1, 1)


class PerformanceFeatureTests(unittest.TestCase):
    def test_arcface_changes_only_the_target_logit(self) -> None:
        classifier = ArcMarginClassifier(2, 2, scale=10.0, margin=0.3)
        with torch.no_grad():
            classifier.weight.copy_(torch.eye(2))
        embedding = torch.tensor([[1.0, 0.0]], requires_grad=True)
        plain = classifier(embedding)
        margin = classifier(embedding, torch.tensor([0]))
        self.assertLess(margin[0, 0], plain[0, 0])
        self.assertAlmostEqual(margin[0, 1].item(), plain[0, 1].item(), places=6)
        margin.sum().backward()
        self.assertTrue(torch.isfinite(embedding.grad).all())

    def test_subcenter_arcface_uses_best_center_per_class(self) -> None:
        classifier = ArcMarginClassifier(2, 2, subcenters=2)
        with torch.no_grad():
            classifier.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
            )
        cosine = classifier.cosine_logits(torch.tensor([[0.0, 1.0]]))
        self.assertEqual(tuple(cosine.shape), (1, 2))
        self.assertAlmostEqual(cosine[0, 0].item(), 1.0, places=5)

    def test_salad_sinkhorn_and_descriptor_have_gradients(self) -> None:
        aggregator = SALADAggregator(8, clusters=4, local_dim=4, global_dim=3)
        scores = torch.randn(2, 5, 20)
        assignments = aggregator._balanced_assignments(scores)
        self.assertTrue(torch.allclose(assignments.sum(dim=1), torch.ones(2, 20), atol=1e-4))
        local = torch.randn(2, 20, 8, requires_grad=True)
        output = aggregator(local, local.mean(dim=1))
        self.assertEqual(tuple(output.shape), (2, 19))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(local.grad).all())

    def test_dolg_orthogonal_fusion(self) -> None:
        aggregator = DOLGAggregator(8, fusion_dim=4)
        local = torch.randn(2, 16, 8, requires_grad=True)
        output = aggregator(local, local.mean(dim=1))
        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertTrue(torch.allclose((output[:, :4] * output[:, 4:]).sum(dim=1), torch.zeros(2), atol=1e-5))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(local.grad).all())

    def test_dinov2_model_uses_dynamic_tokens_and_token_gem(self) -> None:
        captured: dict[str, object] = {}

        def fake_create_model(name: str, **kwargs):
            captured["name"] = name
            captured.update(kwargs)
            return FakeDinoBackbone()

        fake_timm = types.SimpleNamespace(create_model=fake_create_model)
        with mock.patch.dict(sys.modules, {"timm": fake_timm}):
            model = create_model(
                num_classes=3,
                model_name="dinov2_small",
                image_size=378,
                pooling="gem",
                use_projection=True,
                embedding_dim=6,
            )
        embeddings, logits = model(torch.randn(2, 3, 8, 8))
        self.assertEqual(tuple(embeddings.shape), (2, 6))
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-5))
        self.assertEqual(captured["img_size"], 378)
        self.assertIs(captured["dynamic_img_size"], True)
        model.configure_backbone_training(True, train_last_blocks=1)
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.backbone.blocks[0].training)
        self.assertTrue(model.backbone.blocks[1].training)
        self.assertTrue(model.backbone.norm.training)
        model.configure_backbone_training(True, train_last_blocks=0)
        model.train()
        self.assertFalse(
            any(module.training for module in model.backbone.modules())
        )

    def test_supcon_proxy_anchor_and_xbm_are_finite(self) -> None:
        embeddings = F.normalize(torch.randn(4, 6, requires_grad=True), dim=1)
        labels = torch.tensor([0, 0, 1, 1])
        supcon = supervised_contrastive_loss(embeddings, labels, 0.07)
        self.assertTrue(torch.isfinite(supcon))

        proxy = ProxyAnchorLoss(2, 6, alpha=32.0, margin=0.1)
        proxy_value = proxy(embeddings, labels)
        self.assertTrue(torch.isfinite(proxy_value))
        (supcon + proxy_value).backward()
        self.assertTrue(torch.isfinite(proxy.proxies.grad).all())

        memory = CrossBatchMemory(3)
        memory.enqueue(F.normalize(torch.randn(2, 6), dim=1), torch.tensor([0, 1]))
        memory.enqueue(F.normalize(torch.randn(2, 6), dim=1), torch.tensor([2, 3]))
        _, memory_labels = memory.tensors(torch.device("cpu"))
        self.assertEqual(memory_labels.tolist(), [1, 2, 3])
        state = memory.state_dict()
        state["embeddings"] = state["embeddings"].double().requires_grad_()
        state["labels"] = state["labels"].to(torch.int32)
        restored = CrossBatchMemory(3)
        restored.load_state_dict(state)
        self.assertEqual(restored.embeddings.device.type, "cpu")
        self.assertEqual(restored.embeddings.dtype, torch.float32)
        self.assertFalse(restored.embeddings.requires_grad)
        self.assertEqual(restored.labels.dtype, torch.long)
        value = cross_batch_memory_loss(
            embeddings,
            labels,
            *memory.tensors(torch.device("cpu")),
            margin=0.2,
        )
        self.assertTrue(torch.isfinite(value))

    def test_dino_init_rejects_a_different_base_image_size(self) -> None:
        model = torch.nn.Linear(2, 2)
        label_to_index = {"10": 0}
        checkpoint = {
            "model_name": "vit_small_patch14_dinov2.lvd142m",
            "image_size": 70,
            "pooling": "avg",
            "use_projection": False,
            "classifier_type": "linear",
            "model_state": model.state_dict(),
            "label_to_index": label_to_index,
        }
        args = types.SimpleNamespace(
            model_name="vit_small_patch14_dinov2.lvd142m",
            image_size=84,
            pooling="avg",
            use_projection=False,
            classifier="linear",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dino.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "image_size"):
                load_initial_weights(
                    model, path, torch.device("cpu"), label_to_index, args
                )

    def test_hard_validation_passes_paths_and_uses_eval_mode(self) -> None:
        paths = [Path("a.jpg"), Path("b.jpg")]
        records = [
            ImageRecord(paths[0], "10", 0),
            ImageRecord(paths[1], "10", 0),
        ]
        args = types.SimpleNamespace(
            batch_size=2,
            image_size=300,
            val_tta="none",
            input_mean=(0.5, 0.5, 0.5),
            input_std=(0.5, 0.5, 0.5),
            resize_mode="center_crop",
        )
        model = torch.nn.Linear(2, 2)
        fake_embeddings = {
            str(paths[0]): torch.tensor([1.0, 0.0]),
            str(paths[1]): torch.tensor([0.0, 1.0]),
        }
        with mock.patch(
            "triplet_landmark.train.embed_paths",
            return_value=fake_embeddings,
        ) as embed_mock:
            centroids = compute_validation_label_centroids(
                model, records, args, torch.device("cpu")
            )
        self.assertFalse(model.training)
        self.assertEqual(embed_mock.call_args.kwargs["paths"], paths)
        self.assertTrue(torch.isfinite(centroids["10"]).all())

    def test_parse_multiscale_sizes(self) -> None:
        self.assertIsNone(parse_scales(""))
        self.assertEqual(parse_scales("300,384,300"), [300, 384])
        with self.assertRaises(ValueError):
            parse_scales("300,nope")


if __name__ == "__main__":
    unittest.main()
