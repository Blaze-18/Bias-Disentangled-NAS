# Data-Agnostic Bias-Disentangled Structural Embedding Framework
## for Zero-Shot Neural Architecture Search

**Authors:** Student ID 2112576112, Student ID 2110876142  
**Supervisor:** Dr. Md. Ekramul Hamid, University of Rajshahi  
**Course:** CSE4102

---

## Overview

This repository contains the full implementation of the five-stage pipeline described in the
thesis. The framework ranks neural architectures using only structural proxy signals computed
at random initialization. No real input data is required at any point during search.

The five stages are:

1. **Structural Proxy Extraction** — SynFlow, Zen-Score, NASWOT, Parameter Count
2. **Bias Disentanglement** — Regression-based removal of parameter-count-driven bias
3. **PCA Whitening Embedding** — Decorrelated architectural fingerprint construction
4. **Surrogate Training** — Shallow MLP with combined MSE and pairwise ranking loss
5. **Data-Agnostic Inference** — Zero-data ranking of the full search space

---

## Requirements

```
Python >= 3.8
torch >= 1.12
nats_bench
xautodl
scikit-learn >= 1.0
scipy >= 1.7
numpy >= 1.21
tqdm
```

Install all dependencies with:

```bash
pip install nats_bench xautodl torch scikit-learn scipy numpy tqdm
```

---

## Benchmark Data

This project uses the **NATS-Bench Topology Search Space (TSS)** benchmark.

1. Download the file `NATS-tss-v1_0-3ffb9.pickle.pbz2` (1.01 GB) from the
   [NATS-Bench GitHub page](https://github.com/D-X-Y/NATS-Bench).
   The Baidu Pan link with extract code `8duj` is currently the most reliable source.

2. Upload the file to your Google Drive.

3. Set `NATS_BENCH_PATH` in the `CONFIG` dictionary inside `main.py` to the full path
   of the file on your Drive.

---

## Running in Google Colab

```python
# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Clone this repository
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME

# Step 3: Install dependencies
!pip install nats_bench xautodl torch scikit-learn scipy numpy tqdm

# Step 4: Edit the NATS_BENCH_PATH and OUTPUT_DIR in CONFIG inside main.py,
#         then run:
!python main.py
```

---

## Configuration

All settings are controlled through the `CONFIG` dictionary at the top of `main.py`.
Key settings to check before running:

| Key | Default | Description |
|-----|---------|-------------|
| `NATS_BENCH_PATH` | (set this) | Path to the NATS-Bench pickle file |
| `OUTPUT_DIR` | (set this) | Directory where all outputs are saved |
| `NUM_ARCHITECTURES` | 15625 | Number of architectures to process |
| `SURROGATE_TRAIN_SIZE` | 800 | Architectures used to train the surrogate |
| `DATASET` | cifar10-valid | Dataset for ground-truth queries |
| `HP_EPOCHS` | 200 | Training epochs used in benchmark |
| `TOP_K` | 50 | K for Top-K selection accuracy |
| `SEED` | 42 | Random seed for reproducibility |

To run a quick test before full extraction, set `NUM_ARCHITECTURES` to a smaller value
such as 200.

---

## Output Files

All outputs are saved to `OUTPUT_DIR`:

| File | Description |
|------|-------------|
| `ground_truth.npy` | Ground-truth validation accuracies, shape (N,) |
| `proxy_matrix.npy` | Raw proxy scores, shape (N, 4) |
| `proxy_checkpoint.pkl` | Extraction checkpoint (auto-saved every 500 architectures) |
| `bias_artefacts.pkl` | Fitted regression models and param scaler for Stage 2 |
| `pca_artefacts.pkl` | Fitted StandardScaler and PCA for Stage 3 |
| `best_surrogate.pt` | Trained MLP weights (best validation checkpoint) |
| `training_history.pkl` | Per-epoch train and validation loss history |
| `full_ranking.npy` | All architecture indices sorted best to worst |
| `results_summary.pkl` | Complete evaluation results and config |
| `run.log` | Full text log of the entire run |

---

## Resuming Interrupted Runs

Proxy extraction is the longest step. The code automatically saves a checkpoint every
500 architectures to `proxy_checkpoint.pkl`. If the Colab runtime disconnects, simply
re-run `main.py` and extraction will resume from the last checkpoint automatically.

---

## Repository Structure

```
.
|-- main.py          # Complete pipeline (all 5 stages)
|-- requirements.txt # Python dependencies
|-- README.md        # This file
|-- figures
|-- visualize.py
```

---
