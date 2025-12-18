# CV_AD: PatchCore/PaDiM Benchmarking on MVTec AD

This repository provides a single-command workflow to benchmark the PaDiM anomaly detection baseline on the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad). It trains on the MVTec training split, evaluates with image-level AUROC and localization metrics (PRO/AUPRO), and saves a reproducible report plus the worst detected examples.

## Features

- **Single config file**: YAML configuration describing the dataset location, model, and output paths.
- **Reproducibility**: Logs seed, git hash, environment, and dataset hash into `metrics.json`.
- **Outputs**: `metrics.json`, `report.md`, and a `worst_examples/` folder with the top-K anomaly heatmaps.
- **Docker-ready**: The entrypoint is a simple Python CLI (`python -m cvad.cli --config <config.yaml>`), which can be wrapped into a container image.

## Getting Started

1. Ensure the MVTec AD dataset is available locally and update the dataset path in the config file. The expected layout is the standard MVTec structure (e.g., `bottle/train/good/...`, `bottle/test/scratch/...`, `bottle/ground_truth/scratch/...`).
2. Install Python dependencies (PyTorch, torchvision, scikit-learn, PyYAML, Pillow). Example:

   ```bash
   pip install torch torchvision scikit-learn pyyaml pillow
   ```

3. Run the benchmark:

   ```bash
   python -m cvad.cli --config configs/mvtec_padim.yaml
   ```

## Configuration

`configs/mvtec_padim.yaml` shows a runnable template:

```yaml
model: padim
paths:
  output_dir: runs/example

dataset:
  path: /data/mvtec
  category: bottle
  image_size: 256

training:
  seed: 1337
  num_workers: 4
  batch_size: 8
  backbone: wide_resnet50_2
  feature_layers:
    - layer1
    - layer2
    - layer3
  sigma: 4.0

eval:
  worst_k: 16
```

Key fields:

- `dataset.path`: Root of the MVTec dataset.
- `dataset.category`: Specific category (e.g., `bottle`). Set to `null` to iterate over all categories.
- `training`: Seed, batch size, workers, and PaDiM backbone configuration.
- `eval.worst_k`: Number of worst examples exported.
- `paths.output_dir`: Output directory for metrics, report, and worst examples.

## Outputs

After a run you will find:

- `metrics.json` — contains image-level AUROC, pixel-level AUROC, PRO/AUPRO, configuration, environment, git hash, and dataset hash.
- `report.md` — markdown summary of the run and metrics.
- `worst_examples/` — input images, score heatmaps, and masks for the top-K highest anomaly scores.

## Docker

The CLI is self-contained and can be added to a Docker image. Example `Dockerfile` snippet:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install torch torchvision scikit-learn pyyaml pillow
CMD ["python", "-m", "cvad.cli", "--config", "configs/mvtec_padim.yaml"]
```

## Notes

- The implementation currently targets PaDiM. The code structure makes it straightforward to add PatchCore by adding another model implementation and switching via the `model` field.
- PRO/AUPRO metrics are only computed when ground-truth masks are available in the dataset.
