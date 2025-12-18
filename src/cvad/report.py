from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .metrics import MetricResult
from .utils import ensure_dir


def _save_worst_examples(paths: List[str], scores: np.ndarray, masks: Optional[np.ndarray], score_maps: np.ndarray, output_dir: Path, k: int) -> None:
    if k <= 0:
        return
    ensure_dir(output_dir)
    order = scores.argsort()[::-1][:k]
    for rank, idx in enumerate(order):
        img = Image.open(paths[idx]).convert("RGB")
        img.save(output_dir / f"{rank:03d}_input.png")
        heatmap = score_maps[idx]
        heatmap_img = Image.fromarray(((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6) * 255).astype("uint8"))
        heatmap_img.save(output_dir / f"{rank:03d}_score.png")
        if masks is not None:
            mask = masks[idx]
            mask_img = Image.fromarray((mask * 255).astype("uint8"))
            mask_img.save(output_dir / f"{rank:03d}_mask.png")


def render_report(metrics: MetricResult, config: Dict, worst_examples_dir: Path) -> str:
    lines = ["# MVTec AD Benchmark", "", "## Metrics"]
    lines.append(f"- Image-level AUROC: {metrics.image_auroc:.4f}")
    if metrics.pixel_auroc is not None:
        lines.append(f"- Pixel-level AUROC: {metrics.pixel_auroc:.4f}")
    if metrics.pro is not None and metrics.aupro is not None:
        lines.append(f"- PRO: {metrics.pro:.4f}")
        lines.append(f"- AUPRO: {metrics.aupro:.4f}")
    lines.append("")
    lines.append("## Configuration")
    for k, v in config.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append(f"Worst-{config['eval']['worst_k']} examples saved to `{worst_examples_dir}`")
    return "\n".join(lines)


def save_worst_examples(paths: List[str], image_scores: np.ndarray, masks: Optional[np.ndarray], score_maps: np.ndarray, output_dir: Path, k: int) -> None:
    _save_worst_examples(paths, image_scores, masks, score_maps, output_dir, k)
