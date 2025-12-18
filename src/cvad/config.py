from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DatasetConfig:
    path: Path
    category: Optional[str] = None
    image_size: int = 256


@dataclass
class TrainingConfig:
    seed: int = 1337
    num_workers: int = 4
    batch_size: int = 8
    backbone: str = "wide_resnet50_2"
    feature_layers: Optional[List[str]] = None
    sigma: float = 4.0


@dataclass
class EvalConfig:
    worst_k: int = 16


@dataclass
class PathsConfig:
    output_dir: Path = Path("runs/latest")


@dataclass
class Config:
    dataset: DatasetConfig
    training: TrainingConfig = TrainingConfig()
    eval: EvalConfig = EvalConfig()
    paths: PathsConfig = PathsConfig()
    model: str = "padim"

    @staticmethod
    def from_yaml(path: Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text())
        dataset = DatasetConfig(**data.get("dataset", {}))
        training_data = data.get("training", {})
        training = TrainingConfig(**training_data)
        eval_cfg = EvalConfig(**data.get("eval", {}))
        paths_cfg = PathsConfig(**data.get("paths", {}))
        model = data.get("model", "padim")
        return Config(dataset=dataset, training=training, eval=eval_cfg, paths=paths_cfg, model=model)

    def to_dict(self) -> dict:
        def _convert(item):
            if isinstance(item, Path):
                return str(item)
            if dataclasses.is_dataclass(item):
                return {k: _convert(v) for k, v in dataclasses.asdict(item).items()}
            if isinstance(item, (list, tuple)):
                return [_convert(v) for v in item]
            return item

        return _convert(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)
