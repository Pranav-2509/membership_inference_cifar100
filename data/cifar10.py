"""CIFAR-10 dataset loading and utilities."""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


def get_cifar10_transforms(train: bool = True) -> transforms.Compose:
    """Standard CIFAR-10 transforms with normalization."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])


def get_cifar10_loaders(
    data_dir: str = "./datasets",
    batch_size: int = 128,
    num_workers: int = 0,
    train_fraction: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for CIFAR-10.

    Args:
        data_dir: Root directory for dataset.
        batch_size: Batch size for loaders.
        num_workers: Number of dataloader workers.
        train_fraction: Fraction of training set to use (for subset experiments).
        seed: Random seed for reproducibility.

    Returns:
        (train_loader, test_loader)
    """
    data_path = Path(data_dir) / "cifar10"
    data_path.mkdir(parents=True, exist_ok=True)

    train_dataset = CIFAR10(
        root=str(data_path),
        train=True,
        download=True,
        transform=get_cifar10_transforms(train=True),
    )
    test_dataset = CIFAR10(
        root=str(data_path),
        train=False,
        download=True,
        transform=get_cifar10_transforms(train=False),
    )

    if train_fraction < 1.0 and seed is not None:
        torch.manual_seed(seed)
        n_train = len(train_dataset)
        n_use = int(n_train * train_fraction)
        train_dataset, _ = random_split(
            train_dataset,
            [n_use, n_train - n_use],
            generator=torch.Generator().manual_seed(seed),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


class CIFAR10Subset(Subset):
    """
    Subset of CIFAR-10 with optional fixed indices.
    Useful for MIA where we need reproducible member/non-member splits.
    """

    def __init__(self, dataset: CIFAR10, indices: list[int]) -> None:
        super().__init__(dataset, indices)
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """Returns (input, target, global_index)."""
        global_idx = self.indices[idx]
        x, y = self.dataset[global_idx]
        return x, y, global_idx
