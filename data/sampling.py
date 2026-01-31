"""Subset sampling for membership inference splits."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR100

from .cifar100 import get_cifar100_transforms


def sample_subset_excluding(
    n_train: int,
    exclude_indices: list[int],
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample `size` indices from [0, n_train) excluding `exclude_indices`.

    Used for single-point MIA: subset does not contain the target point x.
    """
    exclude = set(exclude_indices)
    pool = np.array([i for i in range(n_train) if i not in exclude])
    if len(pool) < size:
        raise ValueError(
            f"Pool size {len(pool)} (after excluding {len(exclude)}) < requested size {size}"
        )
    return rng.choice(pool, size=size, replace=False)


def sample_membership_splits(
    n_train: int = 50_000,
    n_members: int = 25_000,
    n_non_members: int = 25_000,
    n_calibration: int = 5_000,
    seed: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """
    Sample disjoint member / non-member / calibration index sets from training data.

    Used for MIA: members = in training set, non-members = held-out from train,
    calibration = for threshold/calibration if needed.

    Args:
        n_train: Total training set size (CIFAR-100 = 50_000).
        n_members: Number of member indices (used in training).
        n_non_members: Number of non-member indices (held out, used as "test" for MIA).
        n_calibration: Number of calibration indices (optional, for Neyman-Pearson).
        seed: Random seed.

    Returns:
        Dict with keys: 'member_indices', 'non_member_indices', 'calibration_indices'.
    """
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_train)
    rng.shuffle(all_idx)

    n_total = n_members + n_non_members + n_calibration
    if n_total > n_train:
        raise ValueError(
            f"n_members + n_non_members + n_calibration = {n_total} > n_train = {n_train}"
        )

    members = all_idx[:n_members]
    non_members = all_idx[n_members : n_members + n_non_members]
    calibration = all_idx[n_members + n_non_members : n_members + n_non_members + n_calibration]

    return {
        "member_indices": members,
        "non_member_indices": non_members,
        "calibration_indices": calibration,
    }


def get_mia_splits(
    data_dir: str = "./datasets",
    n_members: int = 25_000,
    n_non_members: int = 10_000,
    n_calibration: int = 5_000,
    seed: Optional[int] = None,
) -> dict:
    """
    Build train (members-only), non-member, and calibration subsets for MIA.

    Returns datasets (as Subsets) and indices for each split.
    """
    from pathlib import Path

    data_path = Path(data_dir) / "cifar100"
    data_path.mkdir(parents=True, exist_ok=True)

    full_train = CIFAR100(
        root=str(data_path),
        train=True,
        download=True,
        transform=get_cifar100_transforms(train=False),  # no augment for scoring
    )

    splits = sample_membership_splits(
        n_train=len(full_train),
        n_members=n_members,
        n_non_members=n_non_members,
        n_calibration=n_calibration,
        seed=seed,
    )

    member_subset = Subset(full_train, splits["member_indices"].tolist())
    non_member_subset = Subset(full_train, splits["non_member_indices"].tolist())
    calibration_subset = Subset(full_train, splits["calibration_indices"].tolist())

    return {
        "member_dataset": member_subset,
        "non_member_dataset": non_member_subset,
        "calibration_dataset": calibration_subset,
        "member_indices": splits["member_indices"],
        "non_member_indices": splits["non_member_indices"],
        "calibration_indices": splits["calibration_indices"],
    }


def get_target_point_and_train_dataset(
    data_dir: str = "./datasets",
    target_index: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Load CIFAR-100 train (eval transforms) and a single target point x.

    Used for single-point MIA: fix x, then repeatedly train with/without x.

    Args:
        data_dir: Root for CIFAR-100.
        target_index: Index of x in train set. If None, draw uniformly.
        seed: For sampling target_index when None.

    Returns:
        Dict with:
          - "dataset": CIFAR-100 train Subset wrapper (full dataset).
          - "target_index": int,
          - "x": (1, C, H, W) tensor,
          - "y": (1,) tensor,
          - "n_train": len(dataset).
    """
    data_path = Path(data_dir) / "cifar100"
    data_path.mkdir(parents=True, exist_ok=True)
    full_train = CIFAR100(
        root=str(data_path),
        train=True,
        download=True,
        transform=get_cifar100_transforms(train=False),
    )
    n_train = len(full_train)
    if target_index is None:
        rng = np.random.default_rng(seed)
        target_index = int(rng.integers(0, n_train))
    else:
        target_index = int(target_index)
    if target_index < 0 or target_index >= n_train:
        raise ValueError(f"target_index {target_index} out of [0, {n_train})")

    x, y = full_train[target_index]
    x_b = x.unsqueeze(0)
    y_b = torch.tensor([y], dtype=torch.long)
    return {
        "dataset": full_train,
        "target_index": target_index,
        "x": x_b,
        "y": y_b,
        "n_train": n_train,
    }
