# GalaxyNet-SCPL

GalaxyNet-SCPL classifies galaxy morphology using DINOv3-ConvNeXt-Small and a multi-scale attention adapter (MMA), with supervised training and structural-consistency pseudo-label learning.

## Usage

Run from the project root with Python 3.10 or later:

Set the dataset, split manifest, structural JSON, local backbone weights, and output paths in `configs/scp_pct_1.yaml`. SCP also requires a matching supervised Base checkpoint via `train.initial_checkpoint`.

```bash
python train.py --config configs/scp_pct_1.yaml --device cuda:0
```

`scp_pct_1.yaml` uses 1% labeled data. Each run saves its resolved configuration, logs, and checkpoints under `output.root`.

Evaluate a completed run using the same YAML and its best checkpoint; add `--tta` to enable test-time augmentation:

```bash
python evaluate.py --config configs/scp_pct_1.yaml --checkpoint /path/to/run/checkpoints/best.pt --split test --device cuda:0
```

## Structural parameters

Generate approximate morphology measurements from JPG, JPEG, and PNG images:

```bash
python preprocess/gen_morphology_features.py \
  --input_path /home/flynn/work/datasets/astro/datasets/gzd5_cls_ten_review \
  --output_json /home/flynn/work/datasets/astro/datasets/gzd5_cls_ten_review/structural.json
```

Our generated structural parameters are provided in [`preprocess/structural.json`](preprocess/structural.json).
