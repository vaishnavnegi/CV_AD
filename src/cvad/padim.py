from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


@dataclass
class PaDiMArtifacts:
    mean: torch.Tensor
    covariance: torch.Tensor
    selected_idx: torch.Tensor


class PaDiM(nn.Module):
    def __init__(self, backbone: str = "wide_resnet50_2", feature_layers: List[str] | None = None, sigma: float = 4.0):
        super().__init__()
        self.backbone_name = backbone
        self.feature_layers = feature_layers or ["layer1", "layer2", "layer3"]
        self.model = self._build_backbone(backbone, self.feature_layers)
        self.sigma = sigma
        self.artifacts: PaDiMArtifacts | None = None

    def _build_backbone(self, backbone: str, layers: List[str]) -> nn.Module:
        backbone_fn = getattr(models, backbone)
        encoder = backbone_fn(weights="IMAGENET1K_V2")
        return create_feature_extractor(encoder, {layer: layer for layer in layers})

    @torch.no_grad()
    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        features = self.model(x)
        maps = [f for f in features.values()]
        for i in range(1, len(maps)):
            if maps[i].shape[-2:] != maps[0].shape[-2:]:
                maps[i] = torch.nn.functional.interpolate(maps[i], size=maps[0].shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat(maps, dim=1)

    @torch.no_grad()
    def fit(self, loader: torch.utils.data.DataLoader, device: torch.device) -> None:
        features = []
        for batch in loader:
            images = batch.image.to(device)
            feats = self._extract(images)
            features.append(feats.cpu())
        feats = torch.cat(features, dim=0)
        b, c, h, w = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(b * h * w, c)
        # PaDiM subsamples channels to reduce covariance size
        n_channels = min(c, 100)
        idx = torch.randperm(c)[:n_channels]
        selected = feats[:, idx]
        mean = selected.mean(dim=0)
        cov = torch.cov(selected.T) + torch.eye(n_channels) * 1e-6
        self.artifacts = PaDiMArtifacts(mean=mean, covariance=cov, selected_idx=idx)

    @torch.no_grad()
    def predict(self, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, List[str]]:
        assert self.artifacts is not None, "Model must be fit before prediction"
        mvn_mean = self.artifacts.mean.to(device)
        mvn_cov = self.artifacts.covariance.to(device)
        inv_cov = torch.linalg.inv(mvn_cov)
        score_maps: List[np.ndarray] = []
        image_scores: List[float] = []
        paths: List[str] = []
        for batch in loader:
            images = batch.image.to(device)
            feats = self._extract(images)
            feats = feats[:, self.artifacts.selected_idx]
            b, c, h, w = feats.shape
            feats = feats.permute(0, 2, 3, 1).reshape(b * h * w, c)
            diff = feats - mvn_mean
            dist = torch.sqrt((diff @ inv_cov * diff).sum(dim=1)).reshape(b, h, w)
            dist = torch.nn.functional.gaussian_blur(dist.unsqueeze(1), kernel_size=9, sigma=self.sigma).squeeze(1)
            score_maps.append(dist.cpu().numpy())
            image_scores.extend(dist.amax(dim=(1, 2)).cpu().tolist())
            paths.extend([str(p) for p in batch.path])
        return np.concatenate(score_maps, axis=0), np.array(image_scores), paths

    def artifacts_state(self) -> Dict[str, torch.Tensor]:
        assert self.artifacts is not None
        return {"mean": self.artifacts.mean, "covariance": self.artifacts.covariance, "selected_idx": self.artifacts.selected_idx}

    def load_artifacts(self, state: Dict[str, torch.Tensor]) -> None:
        self.artifacts = PaDiMArtifacts(mean=state["mean"], covariance=state["covariance"], selected_idx=state["selected_idx"])
