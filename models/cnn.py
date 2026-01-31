"""CNN architecture for CIFAR-10 / CIFAR-100 (32x32x3, configurable classes)."""

from typing import Optional

import torch
import torch.nn as nn


class CIFAR10CNN(nn.Module):
    """
    Small CNN for CIFAR-10/100 (32x32x3 -> num_classes).
    Suitable for membership inference experiments.
    """

    def __init__(
        self,
        num_classes: int = 100,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and pooled features (for MIA or analysis)."""
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits, feat


def get_cifar10_cnn(
    num_classes: int = 100,
    dropout: float = 0.3,
    pretrained_path: Optional[str] = None,
) -> CIFAR10CNN:
    """Construct CIFAR-10/100 CNN, optionally load checkpoint."""
    model = CIFAR10CNN(num_classes=num_classes, dropout=dropout)
    if pretrained_path:
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model
