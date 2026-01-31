"""Training loop for CIFAR-10 models."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    desc: Optional[str] = None,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc=desc or "Train", leave=False):
        inputs, targets = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        n += inputs.size(0)
    return total_loss / n if n else 0.0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model. Returns (mean_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            n += inputs.size(0)
    return (total_loss / n if n else 0.0), (correct / n if n else 0.0)


class Trainer:
    """Simple trainer for CIFAR-10."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        lr: float = 1e-2,
        weight_decay: float = 5e-4,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    def run(
        self,
        epochs: int,
        save_path: Optional[str] = None,
        eval_every: int = 1,
    ) -> list[dict]:
        """Train for `epochs`, optionally save best checkpoint. Returns history."""
        device = self.device
        history = []
        best_acc = -1.0

        for ep in range(epochs):
            loss = train_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                device,
            )
            if self.scheduler is not None:
                self.scheduler.step()

            rec = {"epoch": ep + 1, "train_loss": loss}
            if (ep + 1) % eval_every == 0:
                test_loss, test_acc = evaluate(
                    self.model,
                    self.test_loader,
                    self.criterion,
                    device,
                )
                rec["test_loss"] = test_loss
                rec["test_acc"] = test_acc
                if test_acc > best_acc and save_path:
                    best_acc = test_acc
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
            history.append(rec)
        return history
