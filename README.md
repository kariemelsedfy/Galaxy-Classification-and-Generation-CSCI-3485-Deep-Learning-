# Galaxy Classification and Generation — CSCI 3485

Nathan Bukowski-Thall, Karim ElSedfy, Riley Nelson

## What's in this repo

| File | Description |
|---|---|
| `data_pipeline.ipynb` | Loads the dataset, derives class labels, builds PyTorch DataLoaders. Run this first to make sure everything works locally. |
| `baseline_classifier.ipynb` | Transfer learning with ResNet50. Two-phase training: frozen head first, then full fine-tune. Saves best model to `best_model.pth`. |
| `requirements.txt` | Python dependencies |

## Dataset setup

The image folders and zips are **not tracked in git** — they're too large. You need to get them separately (OneDrive).

Once you have them, your directory should look like this:
```
GALAXY_DATASET/
    final_15000.csv
    IMAGES FILTERED/
Images Differing sizes/
    128/
    224/
```

## Getting started

```bash
pip install -r requirements.txt
```

Then open `data_pipeline.ipynb` and run all cells top to bottom. The last cell shows a grid of labeled galaxy images — if that works, your setup is good.

## Labels

We use 4 classes derived from the Galaxy Zoo 2 crowd-sourced probability scores in the CSV:

| Label | Meaning |
|---|---|
| `elliptical` | Smooth, round galaxy |
| `spiral` | Disk galaxy with spiral arms, no bar |
| `barred_spiral` | Spiral galaxy with a central bar |
| `edge_on` | Disk galaxy viewed edge-on |

Rows where the crowd was split (neither smooth nor featured scored ≥ 0.5) are dropped. That leaves ~13,100 usable images out of 14,990.

## Training (run on HPC)

Open `baseline_classifier.ipynb` on the school HPC (RTX 3080). Before running, change `num_workers=0` to `num_workers=4` in the DataLoader cells — that's only set to 0 to avoid a Jupyter multiprocessing bug on local machines.

Training runs in two phases:
1. **Phase 1 (5 epochs)** — backbone frozen, only the 4-class head trains
2. **Phase 2 (10 epochs)** — full model unfreezes, backbone uses lr=1e-4, head uses lr=1e-3

Best checkpoint is saved to `best_model.pth` (gitignored). The final cells plot loss/accuracy curves and a confusion matrix on the test set.

## Task assignments (from proposal)

- **Nathan** — Experiment 1: compare multiple transfer learning models
- **Karim** — Experiment 2: robustness to image degradation (noise, blur, resolution)
- **Riley** — Experiment 3: Grad-CAM heatmaps
- **All** — GAN generation (time permitting)
