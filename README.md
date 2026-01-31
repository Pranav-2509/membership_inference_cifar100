# Single-Point Membership Inference Attack on CIFAR-100

A PyTorch implementation of a **membership inference attack (MIA)** that determines whether a specific datapoint was in a model’s training set. The pipeline uses repeated “with/without” training, builds loss distributions for a target point, and applies a **Cross Entropy Loss** decision rule for an optimal threshold at a fixed false positive rate.

---

## Overview

**Membership inference:** Given a trained model and a candidate datapoint \(x\), the goal is to infer whether \(x\) was in the model’s training set. A common signal is the **loss** of \(x\) under the model: members tend to have lower loss than non-members.

This repository implements a **single-point MIA** on **CIFAR-100**:

1. **Fix a target point** \(x\) from the CIFAR-100 training set.
2. **Repeat** \(l\) times:
   - Sample \(n\) training points (excluding \(x\)).
   - Train one CNN on these \(n\) points (**without** \(x\)).
   - Train another CNN on the same \(n\) points **plus** \(x\).
   - Record the cross-entropy loss of \(x\) under both models (member vs non-member).
3. **Estimate** the distributions of these losses (histograms + Gaussian fits).
4. **Loss threshold:** Choose a threshold on the loss that fixes the false positive rate (FPR) at \(\alpha\) and maximizes power (true positive rate).
5. **Evaluate:** Train two new held-out models (with/without \(x\)) and check that the rule correctly predicts membership.

---

## Features

- **CIFAR-100** dataset (50k train, 10k test, 100 classes).
- **Single-point MIA:** one target point \(x\), repeated with/without training runs.
- **Loss-based scoring:** higher loss ⇒ more likely non-member.
- **Decision threshold:** control FPR at \(\alpha\); report power and threshold.
- **Visualization:** normalized histograms of member vs non-member losses with fitted Gaussians.
- **Config-driven:** YAML configs for iterations, subset size, epochs, \(\alpha\), seed, etc.
- **Progress bars:** iteration, epoch, and batch-level progress (tqdm).

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+ and torchvision
- See `requirements.txt` for full dependencies.

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/membership_inference_cifar10.git
cd membership_inference_cifar10

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Single-point MIA (main experiment)

From the **project root** (the `membership_inference_cifar10` directory):

```bash
# Default config: configs/single_point_mia.yaml
python run_single_point_mia.py

# Custom config (e.g. quick run for debugging)
python run_single_point_mia.py --config configs/single_point_mia_quick.yaml

# Override paths and seed
python run_single_point_mia.py --config configs/single_point_mia.yaml \
  --data-dir ./datasets --output-dir ./outputs --seed 123
```

**CLI options:**

| Option        | Default                    | Description                    |
|---------------|----------------------------|--------------------------------|
| `--config`    | `configs/single_point_mia.yaml` | Path to YAML config.       |
| `--data-dir`  | `./datasets`               | Root for CIFAR-100 (downloaded here). |
| `--output-dir`| `./outputs`                | Where to save plots.           |
| `--seed`      | From config                | Random seed override.          |

On first run, CIFAR-100 is downloaded under `datasets/cifar100/`. Results and histograms are written to `outputs/` (e.g. `single_point_mia_histograms.png`).

---

## Configuration

Example config: `configs/single_point_mia.yaml`.

| Key             | Description |
|-----------------|-------------|
| `target_index`  | Index of target point \(x\) in the training set. Use `null` to pick at random. |
| `num_classes`   | Number of classes (100 for CIFAR-100). |
| `n_iterations`  | \(l\): number of “with/without” training rounds. |
| `subset_size`   | \(n\): number of points sampled per round (excluding \(x\)). |
| `epochs`        | \(m\): training epochs per CNN per round. |
| `batch_size`    | Mini-batch size for training. |
| `lr`            | Learning rate. |
| `alpha`         | Target FPR for the Decision threshold. |
| `save_plots`    | Whether to save loss histograms. |
| `seed`          | Random seed. |

- **Without \(x\):** each round trains on \(n\) points.
- **With \(x\):** each round trains on \(n + 1\) points (same \(n\) plus \(x\)).

Use `configs/single_point_mia_quick.yaml` for a fast run (few iterations, small subset, few epochs).

---

## Project Structure

```
membership_inference_cifar10/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── run_single_point_mia.py   # CLI entrypoint for single-point MIA
├── __init__.py               # Package version
├── configs/
│   ├── single_point_mia.yaml       # Main single-point MIA config
│   ├── single_point_mia_quick.yaml # Quick/debug config
│   ├── default.yaml                # General MIA experiment config
│   ├── quick.yaml
│   └── repeated.yaml
├── data/
│   ├── cifar10.py            # CIFAR-10 loaders (legacy)
│   ├── cifar100.py           # CIFAR-100 loaders and transforms
│   └── sampling.py           # Subset sampling, get_target_point_and_train_dataset, get_mia_splits
├── experiments/
│   ├── run_single_point_mia.py  # Single-point MIA pipeline
│   └── run_mia.py                # General MIA (train on members, score members vs non-members)
├── models/
│   └── cnn.py                # Small CNN for CIFAR-10/100 (configurable num_classes)
├── training/
│   └── trainer.py            # train_epoch, Trainer
├── stats/
│   ├── distributions.py      # Fit Gaussians/KDE to loss scores
│   └── neyman_pearson.py      # Neyman–Pearson threshold, ROC
├── plots/
│   └── visualization.py      # Histograms, ROC, precision-recall
├── datasets/                 # CIFAR-10/100 downloaded here (git-ignored)
└── outputs/                  # Plots and results (git-ignored)
```

---

## Methodology

### Loss as membership signal

- **Member:** \(x\) was in the training set → model has seen \(x\) → typically **lower** loss on \(x\).
- **Non-member:** \(x\) was not in the training set → typically **higher** loss on \(x\).

So we use **loss** as the MIA score and treat **higher loss ⇒ more likely non-member**.

### Decision rule

We fix the **type-I error** (false positive rate) at \(\alpha\):  
\(P(\text{say member} \mid \text{non-member}) = \alpha\).

With loss as score and “say member” when loss \(< t\):

- Choose threshold \(t\) so that the proportion of **non-member** losses below \(t\) is \(\alpha\) (e.g. empirical \(\alpha\)-quantile of non-member losses).
- **Power** (true positive rate) is the proportion of **member** losses below \(t\).

This gives a threshold that controls FPR and maximizes power under the fitted (e.g. Gaussian) model.

### Evaluation

After computing the threshold from the repeated runs, we train **two new** CNNs (one with \(x\) in the train set, one without) and compute the loss of \(x\) under each. We predict “member” when loss \(< \text{threshold}\) and report whether both predictions are correct.

---

## Output

- **Console:** Target index, threshold, power, FPR, mean losses over iterations, and evaluation summary (loss when \(x\) in/not in train, predicted member?, correct?).
- **Plots:** `outputs/single_point_mia_histograms.png` — normalized histograms of member vs non-member losses with fitted Gaussians (if `save_plots: true`).

---


## References

- Shokri et al., “Membership Inference Attacks Against Machine Learning Models,” IEEE S&P 2017.
- Yeom et al., “Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting,” CSF 2018.
