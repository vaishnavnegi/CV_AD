from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import MVTecDataset, build_transforms, dataset_hash
from .metrics import evaluate
from .padim import PaDiM
from .report import render_report, save_worst_examples
from .utils import ensure_dir, environment_info, git_hash, save_json, set_seed, timestamp


def _prepare_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    transform, mask_transform = build_transforms(cfg.dataset.image_size)
    train_ds = MVTecDataset(cfg.dataset.path, split="train", category=cfg.dataset.category, transform=transform)
    test_ds = MVTecDataset(cfg.dataset.path, split="test", category=cfg.dataset.category, transform=transform, mask_transform=mask_transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)
    return train_loader, test_loader


def _collect_masks(test_ds: MVTecDataset) -> np.ndarray | None:
    masks = []
    for sample in test_ds:
        if sample.mask is None:
            h, w = sample.image.shape[1:]
            masks.append(np.zeros((h, w), dtype=np.float32))
        else:
            masks.append(sample.mask.squeeze(0).numpy())
    return np.stack(masks)


def _serialize_config(cfg: Config) -> Dict[str, Any]:
    return cfg.to_dict()


def run(config_path: Path) -> None:
    cfg = Config.from_yaml(config_path)
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = _prepare_dataloaders(cfg)
    if cfg.model.lower() != "padim":
        raise ValueError(f"Only PaDiM is currently supported. Requested: {cfg.model}")

    model = PaDiM(backbone=cfg.training.backbone, feature_layers=cfg.training.feature_layers, sigma=cfg.training.sigma)
    model.to(device)

    model.fit(train_loader, device)
    score_maps, image_scores, paths = model.predict(test_loader, device)
    labels = np.array([item.is_anomaly for item in test_loader.dataset.items])
    masks = _collect_masks(test_loader.dataset)

    metrics = evaluate(score_maps, image_scores, labels, masks)

    run_dir = Path(cfg.paths.output_dir)
    ensure_dir(run_dir)
    worst_dir = run_dir / "worst_examples"
    save_worst_examples(paths, image_scores, masks, score_maps, worst_dir, cfg.eval.worst_k)

    meta = {
        "timestamp": timestamp(),
        "git_hash": git_hash(),
        "environment": environment_info(),
        "dataset_hash": dataset_hash(cfg.dataset.path),
        "model": cfg.model,
    }

    save_json({"metrics": metrics.to_dict(), "metadata": meta, "config": _serialize_config(cfg)}, run_dir / "metrics.json")
    report_md = render_report(metrics, _serialize_config(cfg), worst_dir)
    (run_dir / "report.md").write_text(report_md)
    print(f"Run complete. Results stored in {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchCore/PaDiM benchmarking on MVTec AD")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
