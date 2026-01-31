"""Membership inference attack experiments."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import get_mia_splits
from ..models import get_cifar10_cnn
from ..training import Trainer


def compute_loss_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[torch.nn.Module] = None,
) -> np.ndarray:
    """
    Compute per-sample cross-entropy loss (MIA score).
    Higher loss -> more likely non-member.
    """
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            scores.append(loss.cpu().numpy())
    return np.concatenate(scores, axis=0)


def compute_confidence_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Compute max softmax probability (MIA score).
    Higher confidence -> more likely member.
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            conf, _ = probs.max(dim=1)
            scores.append(conf.cpu().numpy())
    return np.concatenate(scores, axis=0)


def collect_mia_scores(
    model: torch.nn.Module,
    member_loader: DataLoader,
    non_member_loader: DataLoader,
    device: torch.device,
    score_type: str = "loss",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect MIA scores for members and non-members.

    Args:
        model: Trained target model.
        member_loader: DataLoader over member samples.
        non_member_loader: DataLoader over non-member samples.
        device: Device.
        score_type: 'loss' (higher = non-member) or 'confidence' (higher = member).

    Returns:
        (member_scores, non_member_scores)
    """
    if score_type == "loss":
        member_scores = compute_loss_scores(model, member_loader, device)
        non_member_scores = compute_loss_scores(model, non_member_loader, device)
    else:
        member_scores = compute_confidence_scores(model, member_loader, device)
        non_member_scores = compute_confidence_scores(model, non_member_loader, device)
    return member_scores, non_member_scores


def run_mia_experiment(
    config: dict,
    data_dir: str = "./datasets",
    output_dir: str = "./outputs",
    seed: Optional[int] = None,
) -> dict:
    """
    Run full MIA experiment: train on members, score members vs non-members.

    Config keys: n_members, n_non_members, n_calibration, epochs, batch_size,
                 lr, score_type, save_model.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_members = config.get("n_members", 25_000)
    n_non_members = config.get("n_non_members", 10_000)
    n_calibration = config.get("n_calibration", 5_000)
    epochs = config.get("epochs", 30)
    batch_size = config.get("batch_size", 128)
    lr = config.get("lr", 0.01)
    score_type = config.get("score_type", "loss")
    save_model = config.get("save_model", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    splits = get_mia_splits(
        data_dir=data_dir,
        n_members=n_members,
        n_non_members=n_non_members,
        n_calibration=n_calibration,
        seed=seed,
    )

    member_loader = DataLoader(
        splits["member_dataset"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    non_member_loader = DataLoader(
        splits["non_member_dataset"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    cal_loader = DataLoader(
        splits["calibration_dataset"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = get_cifar10_cnn()
    save_path = str(Path(output_dir) / "best_model.pt") if save_model else None

    trainer = Trainer(
        model=model,
        train_loader=member_loader,
        test_loader=non_member_loader,
        device=device,
        lr=lr,
    )
    history = trainer.run(epochs=epochs, save_path=save_path, eval_every=1)

    if save_model and save_path:
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

    member_scores, non_member_scores = collect_mia_scores(
        model, member_loader, non_member_loader, device, score_type=score_type
    )

    calibration_scores = None
    if n_calibration > 0:
        calibration_scores = compute_loss_scores(model, cal_loader, device) if score_type == "loss" else compute_confidence_scores(model, cal_loader, device)

    results = {
        "member_scores": member_scores,
        "non_member_scores": non_member_scores,
        "calibration_scores": calibration_scores,
        "score_type": score_type,
        "history": history,
        "config": config,
    }
    return results
