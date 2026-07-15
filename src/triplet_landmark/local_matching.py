from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


class LightGlueGeometricReranker:
    """Optional pair reranker using local matches and homography inliers."""

    def __init__(
        self,
        device: torch.device,
        features: str = "aliked",
        max_keypoints: int = 2048,
        max_cached_images: int = 32,
    ) -> None:
        try:
            import cv2
            from lightglue import ALIKED, DISK, LightGlue, SuperPoint
            from lightglue.utils import load_image, rbd
        except ImportError as exc:
            raise ImportError(
                "LightGlue reranking is optional. Install it with: "
                "python -m pip install -r requirements-lightglue.txt"
            ) from exc

        extractors = {
            "aliked": ALIKED,
            "disk": DISK,
            "superpoint": SuperPoint,
        }
        if features not in extractors:
            raise ValueError(f"Unsupported LightGlue features: {features}")
        self.device = device
        self.cv2 = cv2
        self.load_image = load_image
        self.rbd = rbd
        self.extractor = extractors[features](
            max_num_keypoints=max_keypoints
        ).eval().to(device)
        self.matcher = LightGlue(features=features).eval().to(device)
        if max_cached_images <= 0:
            raise ValueError("max_cached_images must be positive.")
        self.max_cached_images = max_cached_images
        self._feature_cache: OrderedDict[str, dict[str, Any]] = (
            OrderedDict()
        )

    @torch.inference_mode()
    def _features(self, path: Path) -> dict[str, Any]:
        key = str(path.resolve())
        if key in self._feature_cache:
            features = self._feature_cache.pop(key)
            self._feature_cache[key] = features
            return features
        image = self.load_image(key).to(self.device)
        features = self.extractor.extract(image)
        self._feature_cache[key] = features
        while len(self._feature_cache) > self.max_cached_images:
            self._feature_cache.popitem(last=False)
        return features

    @torch.inference_mode()
    def score(self, first_path: Path, second_path: Path) -> float:
        first = self._features(first_path)
        second = self._features(second_path)
        matches = self.matcher({"image0": first, "image1": second})
        first_unbatched, second_unbatched, matches_unbatched = [
            self.rbd(value) for value in (first, second, matches)
        ]
        match_indexes = matches_unbatched["matches"]
        if match_indexes.shape[0] < 4:
            return 0.0
        points_first = first_unbatched["keypoints"][match_indexes[:, 0]]
        points_second = second_unbatched["keypoints"][match_indexes[:, 1]]
        method = getattr(self.cv2, "USAC_MAGSAC", self.cv2.RANSAC)
        _, inlier_mask = self.cv2.findHomography(
            points_first.detach().float().cpu().numpy(),
            points_second.detach().float().cpu().numpy(),
            method,
            3.0,
        )
        if inlier_mask is None:
            return 0.0
        return float(inlier_mask.reshape(-1).mean())
