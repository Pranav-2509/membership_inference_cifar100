"""Single-point membership inference attack on CIFAR-100.

This experiment implements the following procedure:

1. Take the CIFAR-100 training set.
2. Select one target datapoint ``x`` (by index or at random).
3. For ``l = n_iterations`` rounds:
   - Sample ``n = subset_size`` datapoints from the *remaining* training set
     (excluding ``x``).
   - Train one CNN on these ``n`` points (without ``x``) for ``m = epochs``
     epochs.
   - Train another CNN on the same ``n`` points **plus** ``x`` for ``m`` epochs.
   - Record the cross-entropy loss of ``x`` under:
       * the model where ``x`` was *not* in the training set (non-member loss),
       * the model where ``x`` *was* in the training set (member loss).
4. Build normalized histograms of these two loss distributions, fit Gaussians
   to approximate the underlying probability distributions, and run a
   Neyman–Pearson test to obtain an optimal decision threshold on the loss.
5. Finally, train *two new* CNNs (one with ``x`` in the train set and one
   without) and use the learned threshold to decide in which model ``x`` was
   a member; we expect to classify both cases correctly with high probability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..data import get_target_point_and_train_dataset, sample_subset_excluding
from ..models import get_cifar10_cnn
from ..training import train_epoch
from ..plots import plot_score_distributions
from ..stats import fit_score_distributions, neyman_pearson_test


def _make_dataloader(
    dataset: torch.utils.data.Dataset,
    indices: np.ndarray | list[int],
    batch_size: int,
) -> DataLoader:
    """Create a shuffled DataLoader over a subset of ``dataset``."""
    if isinstance(indices, np.ndarray):
        indices = indices.tolist()
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )


def _loss_for_point(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    """Compute per-sample cross-entropy loss for a single point (x, y)."""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        loss = criterion(logits, y.to(device))
    # criterion uses reduction="none" so we get shape (1,)
    return float(loss.view(-1)[0].item())


def _train_model_on_subset(
    dataset: torch.utils.data.Dataset,
    indices: np.ndarray | list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    num_classes: int = 100,
    iter_num: Optional[int] = None,
    label: str = "",
) -> nn.Module:
    """Instantiate and train a CIFAR-100 CNN on the given subset."""
    loader = _make_dataloader(dataset, indices, batch_size=batch_size)
    model = get_cifar10_cnn(num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    criterion = nn.CrossEntropyLoss()

    prefix = f"Iter {iter_num} | {label} | " if iter_num is not None and label else ""
    for ep in tqdm(range(epochs), desc=f"{prefix}Epoch", unit="epoch", leave=False):
        train_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device,
            desc=f"Epoch {ep + 1}/{epochs}",
        )
    return model


def run_single_point_mia(
    config: Dict[str, Any],
    data_dir: str = "./datasets",
    output_dir: str = "./outputs",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the single-point membership inference experiment described above.

    Config keys (with defaults):
        - target_index: int | null   (if null, choose x at random)
        - n_iterations: int          (number of train-with/without-x rounds, l)
        - subset_size: int           (number of points sampled each round, n)
        - epochs: int                (training epochs per model, m)
        - batch_size: int
        - lr: float
        - alpha: float               (target type-I error for NP test)
        - num_classes: int           (100 for CIFAR-100)
        - save_plots: bool
        - seed: int | null           (fallback if function argument seed is None)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters from config
    cfg_target_index = config.get("target_index")
    n_iterations = int(config.get("n_iterations", 100))
    subset_size = int(config.get("subset_size", 1000))
    epochs = int(config.get("epochs", 10))
    batch_size = int(config.get("batch_size", 64))
    lr = float(config.get("lr", 0.01))
    alpha = float(config.get("alpha", 0.05))
    save_plots = bool(config.get("save_plots", True))
    num_classes = int(config.get("num_classes", 100))

    if seed is None:
        seed = config.get("seed")

    # Reproducibility
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1–2. Load CIFAR-100 train set and select target point x
    ds_info = get_target_point_and_train_dataset(
        data_dir=data_dir,
        target_index=cfg_target_index,
        seed=seed,
    )
    dataset = ds_info["dataset"]
    target_index = int(ds_info["target_index"])
    x = ds_info["x"]  # shape (1, C, H, W)
    y = ds_info["y"]  # shape (1,)
    n_train = int(ds_info["n_train"])

    # 3–5. Repeat training with/without x and record losses
    member_losses: list[float] = []  # loss when x IS in the train set
    non_member_losses: list[float] = []  # loss when x is NOT in the train set

    point_loss_criterion = nn.CrossEntropyLoss(reduction="none")

    iter_bar = tqdm(
        range(n_iterations),
        desc="Iteration",
        unit="iter",
        leave=True,
    )
    for it in iter_bar:
        iter_bar.set_postfix_str(f"iter {it + 1}/{n_iterations}")
        # 3. Sample subset from remaining data (exclude x)
        subset_indices = sample_subset_excluding(
            n_train=n_train,
            exclude_indices=[target_index],
            size=subset_size,
            rng=rng,
        )

        # 4. Train CNNs on subset (without x) and subset + x
        # Model where x is NOT in the train set
        model_without_x = _train_model_on_subset(
            dataset=dataset,
            indices=subset_indices,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            num_classes=num_classes,
            iter_num=it + 1,
            label="without x",
        )

        # Model where x IS in the train set
        subset_with_x = np.concatenate([subset_indices, np.array([target_index], dtype=int)])
        model_with_x = _train_model_on_subset(
            dataset=dataset,
            indices=subset_with_x,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            num_classes=num_classes,
            iter_num=it + 1,
            label="with x",
        )

        # 5. Record loss of the point x in both scenarios
        loss_non_member = _loss_for_point(
            model_without_x,
            x,
            y,
            device=device,
            criterion=point_loss_criterion,
        )
        loss_member = _loss_for_point(
            model_with_x,
            x,
            y,
            device=device,
            criterion=point_loss_criterion,
        )

        non_member_losses.append(loss_non_member)
        member_losses.append(loss_member)

    member_losses_arr = np.asarray(member_losses, dtype=np.float32)
    non_member_losses_arr = np.asarray(non_member_losses, dtype=np.float32)

    # 6–7. Plot normalized histograms and approximate distributions
    hist_path = out_dir / "single_point_mia_histograms.png"
    if save_plots:
        plot_score_distributions(
            member_scores=member_losses_arr,
            non_member_scores=non_member_losses_arr,
            score_type="loss",
            fit_normal=True,
            save_path=str(hist_path),
        )

    fitted_member, fitted_non_member = fit_score_distributions(
        member_losses_arr,
        non_member_losses_arr,
        family="normal",
    )

    # 8. Neyman–Pearson decision rule on loss scores
    np_result = neyman_pearson_test(
        member_scores=member_losses_arr,
        non_member_scores=non_member_losses_arr,
        alpha=alpha,
        score_higher_is_non_member=True,  # higher loss => more likely non-member
        fit_family="normal",
    )
    threshold = float(np_result["threshold"])
    power = float(np_result["power"])
    fpr = float(np_result["fpr"])

    # 9–10. Train two new CNNs (one with x in train set, one without) and
    #       check whether the NP rule predicts membership correctly.
    eval_subset_indices = sample_subset_excluding(
        n_train=n_train,
        exclude_indices=[target_index],
        size=subset_size,
        rng=rng,
    )

    # Model A: x IS in the training set
    eval_indices_with_x = np.concatenate(
        [eval_subset_indices, np.array([target_index], dtype=int)]
    )
    model_eval_with_x = _train_model_on_subset(
        dataset=dataset,
        indices=eval_indices_with_x,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        num_classes=num_classes,
    )

    # Model B: x is NOT in the training set
    model_eval_without_x = _train_model_on_subset(
        dataset=dataset,
        indices=eval_subset_indices,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        num_classes=num_classes,
    )

    # Loss of x under the model trained WITH x (should typically be lower)
    loss_when_x_in_train = _loss_for_point(
        model_eval_with_x,
        x,
        y,
        device=device,
        criterion=point_loss_criterion,
    )
    # Loss of x under the model trained WITHOUT x (should typically be higher)
    loss_when_x_not_in_train = _loss_for_point(
        model_eval_without_x,
        x,
        y,
        device=device,
        criterion=point_loss_criterion,
    )

    # Decision rule: say "member" when loss < threshold (lower loss => more likely member)
    say_member_when_x_in_train = loss_when_x_in_train < threshold
    say_member_when_x_not_in_train = loss_when_x_not_in_train < threshold

    correct_when_x_in_train = bool(say_member_when_x_in_train)
    correct_when_x_not_in_train = bool(not say_member_when_x_not_in_train)
    both_correct = correct_when_x_in_train and correct_when_x_not_in_train

    eval_results = {
        "loss_when_x_in_train": float(loss_when_x_in_train),
        "loss_when_x_not_in_train": float(loss_when_x_not_in_train),
        "say_member_when_x_in_train": bool(say_member_when_x_in_train),
        "say_member_when_x_not_in_train": bool(say_member_when_x_not_in_train),
        "correct_when_x_in_train": correct_when_x_in_train,
        "correct_when_x_not_in_train": correct_when_x_not_in_train,
        "both_correct": both_correct,
    }

    return {
        "target_index": target_index,
        "alpha": alpha,
        "threshold": threshold,
        "power": power,
        "fpr": fpr,
        "hist_path": str(hist_path) if save_plots else None,
        "member_losses": member_losses_arr,
        "non_member_losses": non_member_losses_arr,
        "member_dist_params": fitted_member.params,
        "non_member_dist_params": fitted_non_member.params,
        "np_result": np_result,
        "eval": eval_results,
        "config": dict(config),
    }


__all__ = ["run_single_point_mia"]

