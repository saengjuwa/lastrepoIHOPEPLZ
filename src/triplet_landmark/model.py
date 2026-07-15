from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


MODEL_NAME = "tf_efficientnetv2_s.in21k"
BACKBONE_ALIASES = {
    "efficientnetv2_s": MODEL_NAME,
    "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_base": "vit_base_patch14_dinov2.lvd142m",
}


def resolve_backbone_name(name: str) -> str:
    return BACKBONE_ALIASES.get(name, name)


def is_dinov2_backbone(name: str) -> bool:
    return "dinov2" in resolve_backbone_name(name).lower()


class GeMPool(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor([p], dtype=torch.float32))
        self.eps = eps

    def forward(self, local_features: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=self.eps)
        return local_features.clamp(min=self.eps).pow(p).mean(dim=1).pow(1.0 / p)


class SALADAggregator(nn.Module):
    """Optimal-transport local aggregation with a learnable dustbin cluster."""

    def __init__(
        self,
        input_dim: int,
        clusters: int = 16,
        local_dim: int = 64,
        global_dim: int = 256,
        sinkhorn_iterations: int = 3,
    ) -> None:
        super().__init__()
        if clusters <= 0 or local_dim <= 0 or global_dim <= 0:
            raise ValueError("SALAD dimensions and cluster count must be positive.")
        if sinkhorn_iterations <= 0:
            raise ValueError("SALAD Sinkhorn iterations must be positive.")
        self.clusters = clusters
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.sinkhorn_iterations = sinkhorn_iterations
        self.local_projection = nn.Sequential(
            nn.Linear(input_dim, local_dim),
            nn.LayerNorm(local_dim),
            nn.ReLU(inplace=True),
        )
        self.assignment = nn.Linear(input_dim, clusters + 1)
        self.centroids = nn.Parameter(torch.empty(clusters, local_dim))
        nn.init.kaiming_uniform_(self.centroids, a=math.sqrt(5))
        self.global_projection = nn.Sequential(
            nn.Linear(input_dim, global_dim),
            nn.LayerNorm(global_dim),
        )
        self.output_dim = clusters * local_dim + global_dim

    def _balanced_assignments(self, scores: torch.Tensor) -> torch.Tensor:
        # Log-domain Sinkhorn with uniform cluster and patch marginals.
        original_dtype = scores.dtype
        scores = scores.float()
        batch, cluster_count, patch_count = scores.shape
        log_cluster_mass = scores.new_full(
            (batch, cluster_count), -math.log(cluster_count)
        )
        log_patch_mass = scores.new_full(
            (batch, patch_count), -math.log(patch_count)
        )
        row_scale = torch.zeros_like(log_cluster_mass)
        column_scale = torch.zeros_like(log_patch_mass)
        for _ in range(self.sinkhorn_iterations):
            row_scale = log_cluster_mass - torch.logsumexp(
                scores + column_scale[:, None, :], dim=2
            )
            column_scale = log_patch_mass - torch.logsumexp(
                scores + row_scale[:, :, None], dim=1
            )
        transport = torch.exp(
            scores + row_scale[:, :, None] + column_scale[:, None, :]
        )
        return (transport * patch_count).to(original_dtype)

    def forward(
        self,
        local_features: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> torch.Tensor:
        if local_features.size(1) <= self.clusters:
            raise ValueError(
                "SALAD needs more local patches than real clusters; "
                f"got patches={local_features.size(1)} clusters={self.clusters}."
            )
        local = self.local_projection(local_features)
        scores = self.assignment(local_features).transpose(1, 2)
        assignments = self._balanced_assignments(scores)[:, : self.clusters]
        residuals = local[:, None, :, :] - self.centroids[None, :, None, :]
        vlad = (assignments[..., None] * residuals).sum(dim=2)
        vlad = F.normalize(vlad, p=2, dim=2).flatten(1)
        vlad = F.normalize(vlad, p=2, dim=1)
        global_descriptor = F.normalize(
            self.global_projection(global_feature), p=2, dim=1
        )
        return torch.cat([vlad, global_descriptor], dim=1)


class DOLGAggregator(nn.Module):
    """Multi-dilation local attention with orthogonal local/global fusion."""

    def __init__(self, input_dim: int, fusion_dim: int = 512) -> None:
        super().__init__()
        if fusion_dim <= 0:
            raise ValueError("DOLG fusion dimension must be positive.")
        self.local_reduce = nn.Conv2d(input_dim, fusion_dim, kernel_size=1, bias=False)
        self.local_context = nn.ModuleList(
            [
                nn.Conv2d(
                    fusion_dim,
                    fusion_dim,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=fusion_dim,
                    bias=False,
                )
                for dilation in (1, 2, 3)
            ]
        )
        self.local_mix = nn.Conv2d(fusion_dim * 3, fusion_dim, kernel_size=1)
        hidden_dim = max(1, fusion_dim // 4)
        self.local_attention = nn.Sequential(
            nn.Conv2d(fusion_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Softplus(),
        )
        self.global_pool = GeMPool()
        self.global_projection = nn.Linear(input_dim, fusion_dim)
        self.output_dim = fusion_dim * 2

    def forward(
        self,
        local_features: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> torch.Tensor:
        del global_feature
        patch_count = local_features.size(1)
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise ValueError(
                "DOLG needs a square CNN feature map; "
                f"got {patch_count} flattened locations."
            )
        feature_map = local_features.transpose(1, 2).reshape(
            local_features.size(0), local_features.size(2), side, side
        )
        reduced = self.local_reduce(feature_map)
        contextual = torch.cat(
            [F.silu(layer(reduced)) for layer in self.local_context], dim=1
        )
        local_map = self.local_mix(contextual)
        attention = self.local_attention(local_map).clamp_min(1e-6)
        local = (attention * local_map).flatten(2).sum(dim=2)
        local = local / attention.flatten(2).sum(dim=2)
        global_descriptor = self.global_projection(self.global_pool(local_features))
        denominator = global_descriptor.square().sum(dim=1, keepdim=True).clamp_min(1e-6)
        projection = (local * global_descriptor).sum(dim=1, keepdim=True)
        orthogonal_local = local - projection / denominator * global_descriptor
        return torch.cat(
            [F.normalize(global_descriptor, dim=1), F.normalize(orthogonal_local, dim=1)],
            dim=1,
        )


class ArcMarginClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.3,
        subcenters: int = 1,
    ) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError("ArcFace scale must be positive.")
        if not 0.0 <= margin < math.pi / 2:
            raise ValueError("ArcFace margin must be in [0, pi/2).")
        if subcenters <= 0:
            raise ValueError("ArcFace subcenters must be positive.")
        self.num_classes = num_classes
        self.subcenters = subcenters
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes * subcenters, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def cosine_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))
        cosine = cosine.view(-1, self.num_classes, self.subcenters)
        return cosine.max(dim=2).values.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    def class_proxies(self) -> torch.Tensor:
        centers = F.normalize(self.weight, dim=1).view(
            self.num_classes, self.subcenters, -1
        )
        return F.normalize(centers.mean(dim=1), dim=1)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cosine = self.cosine_logits(embeddings)
        if labels is None:
            return cosine * self.scale
        target_cosine = cosine.gather(1, labels[:, None])
        target_sine = torch.sqrt((1.0 - target_cosine.square()).clamp_min(1e-6))
        phi = target_cosine * math.cos(self.margin) - target_sine * math.sin(self.margin)
        threshold = math.cos(math.pi - self.margin)
        phi = torch.where(
            target_cosine > threshold,
            phi,
            target_cosine - math.sin(math.pi - self.margin) * self.margin,
        )
        logits = cosine.clone()
        logits.scatter_(1, labels[:, None], phi)
        return logits * self.scale


class LandmarkEmbeddingNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool,
        embedding_dim: int = 512,
        pooling: str = "avg",
        gem_p: float = 3.0,
        use_projection: bool = False,
        model_name: str = MODEL_NAME,
        image_size: int = 300,
        classifier_type: str = "linear",
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.3,
        subcenters: int = 3,
        salad_clusters: int = 16,
        salad_local_dim: int = 64,
        salad_global_dim: int = 256,
        sinkhorn_iterations: int = 3,
        dolg_dim: int = 512,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm with: pip install -r requirements.txt") from exc

        if pooling not in {"avg", "gem", "salad", "dolg"}:
            raise ValueError("pooling must be avg, gem, salad, or dolg.")
        if classifier_type not in {"linear", "arcface", "subcenter_arcface"}:
            raise ValueError("Unsupported classifier type.")
        self.model_name = resolve_backbone_name(model_name)
        if is_dinov2_backbone(self.model_name) and image_size % 14:
            raise ValueError("DINOv2 image size must be divisible by its patch size (14).")
        self.pooling = pooling
        self.use_projection = use_projection
        self.classifier_type = classifier_type
        self.requires_labels_for_logits = classifier_type != "linear"

        create_kwargs: dict[str, object] = {
            "pretrained": pretrained,
            "num_classes": 0,
            "global_pool": "",
        }
        if is_dinov2_backbone(self.model_name):
            create_kwargs["img_size"] = image_size
            create_kwargs["dynamic_img_size"] = True
        self.backbone = timm.create_model(self.model_name, **create_kwargs)
        self._backbone_training_restricted = False
        self._train_last_blocks = 0
        feature_dim = int(self.backbone.num_features)

        if pooling == "gem":
            self.pool = GeMPool(gem_p)
            pooled_dim = feature_dim
        elif pooling == "salad":
            self.pool = SALADAggregator(
                feature_dim,
                clusters=salad_clusters,
                local_dim=salad_local_dim,
                global_dim=salad_global_dim,
                sinkhorn_iterations=sinkhorn_iterations,
            )
            pooled_dim = self.pool.output_dim
        elif pooling == "dolg":
            self.pool = DOLGAggregator(feature_dim, fusion_dim=dolg_dim)
            pooled_dim = self.pool.output_dim
        else:
            self.pool = nn.Identity()
            pooled_dim = feature_dim

        if use_projection:
            self.embedding_dim = embedding_dim
            self.projection = nn.Sequential(
                nn.Linear(pooled_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.SiLU(inplace=True),
            )
        else:
            self.embedding_dim = pooled_dim
            self.projection = nn.Identity()

        if classifier_type == "linear":
            self.classifier: nn.Module = nn.Linear(self.embedding_dim, num_classes)
        else:
            centers = subcenters if classifier_type == "subcenter_arcface" else 1
            self.classifier = ArcMarginClassifier(
                self.embedding_dim,
                num_classes,
                scale=arcface_scale,
                margin=arcface_margin,
                subcenters=centers,
            )

    def _apply_backbone_training_mode(self) -> None:
        if not self.training:
            return
        if not self._backbone_training_restricted:
            self.backbone.train()
            return
        self.backbone.eval()
        if self._train_last_blocks:
            blocks = list(self.backbone.blocks)
            for block in blocks[-self._train_last_blocks :]:
                block.train()
            norm = getattr(self.backbone, "norm", None)
            if norm is not None:
                norm.train()

    def train(self, mode: bool = True) -> LandmarkEmbeddingNet:
        super().train(mode)
        self._apply_backbone_training_mode()
        return self

    def configure_backbone_training(self, freeze: bool, train_last_blocks: int = 0) -> None:
        if train_last_blocks < 0:
            raise ValueError("train_last_blocks must be non-negative.")
        restricted = freeze or train_last_blocks > 0
        self._backbone_training_restricted = restricted
        self._train_last_blocks = train_last_blocks
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not restricted
        if train_last_blocks:
            blocks = getattr(self.backbone, "blocks", None)
            if blocks is None:
                raise ValueError("This backbone does not expose transformer blocks.")
            blocks = list(blocks)
            if train_last_blocks > len(blocks):
                raise ValueError(
                    f"train_last_blocks={train_last_blocks} exceeds "
                    f"the backbone block count ({len(blocks)})."
                )
            for block in blocks[-train_last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            norm = getattr(self.backbone, "norm", None)
            if norm is not None:
                for parameter in norm.parameters():
                    parameter.requires_grad = True
        self._apply_backbone_training_mode()

    def _feature_parts(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim == 4:
            if (
                features.shape[1] != self.backbone.num_features
                and features.shape[-1] == self.backbone.num_features
            ):
                features = features.permute(0, 3, 1, 2).contiguous()
            local = features.flatten(2).transpose(1, 2)
            return local, local.mean(dim=1)
        if features.ndim == 3:
            prefix_count = int(getattr(self.backbone, "num_prefix_tokens", 1))
            prefix_count = min(prefix_count, max(0, features.size(1) - 1))
            local = features[:, prefix_count:]
            global_feature = (
                features[:, :prefix_count].mean(dim=1)
                if prefix_count
                else local.mean(dim=1)
            )
            return local, global_feature
        if features.ndim == 2:
            return features[:, None, :], features
        raise ValueError(f"Unsupported backbone feature shape: {tuple(features.shape)}")

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        raw_features = self.backbone.forward_features(images)
        if not isinstance(raw_features, torch.Tensor):
            raise TypeError("The selected timm backbone must return a feature tensor.")
        local_features, global_feature = self._feature_parts(raw_features)
        if self.pooling == "avg":
            return global_feature
        if self.pooling == "gem":
            return self.pool(local_features)
        return self.pool(local_features, global_feature)

    def class_proxies(self) -> torch.Tensor:
        if isinstance(self.classifier, ArcMarginClassifier):
            return self.classifier.class_proxies()
        return F.normalize(self.classifier.weight, dim=1)

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.projection(self._extract_features(images))
        embeddings = F.normalize(features, p=2, dim=1)
        if isinstance(self.classifier, ArcMarginClassifier):
            logits = self.classifier(embeddings, labels)
        else:
            logits = self.classifier(features)
        return embeddings, logits


# Backward-compatible name for code that imported the original class.
EfficientNetV2EmbeddingNet = LandmarkEmbeddingNet


def create_model(
    num_classes: int,
    pretrained: bool = False,
    embedding_dim: int = 512,
    pooling: str = "avg",
    gem_p: float = 3.0,
    use_projection: bool = False,
    model_name: str = MODEL_NAME,
    image_size: int = 300,
    classifier_type: str = "linear",
    arcface_scale: float = 30.0,
    arcface_margin: float = 0.3,
    subcenters: int = 3,
    salad_clusters: int = 16,
    salad_local_dim: int = 64,
    salad_global_dim: int = 256,
    sinkhorn_iterations: int = 3,
    dolg_dim: int = 512,
) -> nn.Module:
    return LandmarkEmbeddingNet(
        num_classes=num_classes,
        pretrained=pretrained,
        embedding_dim=embedding_dim,
        pooling=pooling,
        gem_p=gem_p,
        use_projection=use_projection,
        model_name=model_name,
        image_size=image_size,
        classifier_type=classifier_type,
        arcface_scale=arcface_scale,
        arcface_margin=arcface_margin,
        subcenters=subcenters,
        salad_clusters=salad_clusters,
        salad_local_dim=salad_local_dim,
        salad_global_dim=salad_global_dim,
        sinkhorn_iterations=sinkhorn_iterations,
        dolg_dim=dolg_dim,
    )
