from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn import metrics


@dataclass
class MetricResult:
    image_auroc: float
    pixel_auroc: Optional[float]
    pro: Optional[float]
    aupro: Optional[float]

    def to_dict(self) -> Dict[str, float | None]:
        return {
            "image_auroc": self.image_auroc,
            "pixel_auroc": self.pixel_auroc,
            "pro": self.pro,
            "aupro": self.aupro,
        }


def compute_image_level_metrics(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(metrics.roc_auc_score(labels, scores))


def compute_pixel_level_auroc(score_maps: np.ndarray, masks: np.ndarray) -> float:
    scores_flat = score_maps.reshape(score_maps.shape[0], -1)
    masks_flat = masks.reshape(masks.shape[0], -1)
    return float(metrics.roc_auc_score(masks_flat.ravel(), scores_flat.ravel()))


def compute_pro(score_maps: np.ndarray, masks: np.ndarray, max_fpr: float = 0.3) -> tuple[float, float]:
    score_maps = score_maps.reshape(score_maps.shape[0], -1)
    masks = masks.reshape(masks.shape[0], -1)
    thresholds = np.linspace(score_maps.min(), score_maps.max(), num=200)
    pros, fprs = [], []
    total_masks = masks.sum(axis=1) + 1e-8
    for th in thresholds:
        binary = (score_maps > th).astype(np.float32)
        inter = (binary * masks).sum(axis=1)
        union = masks.sum(axis=1)
        pro = inter / total_masks
        fp_pixels = (binary * (1 - masks)).sum(axis=1)
        fpr = fp_pixels / (score_maps.shape[1] - union + 1e-8)
        pros.append(pro.mean())
        fprs.append(fpr.mean())
    pros = np.array(pros)
    fprs = np.array(fprs)
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]
    aupro = float(np.trapz(pros[fprs <= max_fpr], fprs[fprs <= max_fpr]) / max_fpr)
    return float(pros.max()), aupro


def evaluate(score_maps: np.ndarray, image_scores: np.ndarray, labels: np.ndarray, masks: Optional[np.ndarray]) -> MetricResult:
    image_auroc = compute_image_level_metrics(image_scores, labels)
    pixel_auroc = None
    pro = None
    aupro = None
    if masks is not None:
        pixel_auroc = compute_pixel_level_auroc(score_maps, masks)
        pro, aupro = compute_pro(score_maps, masks)
    return MetricResult(image_auroc=image_auroc, pixel_auroc=pixel_auroc, pro=pro, aupro=aupro)
