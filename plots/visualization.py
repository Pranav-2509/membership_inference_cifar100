"""Plotting utilities for MIA experiments."""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from ..stats import roc_curve, neyman_pearson_test, fit_score_distributions


def _default_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10


def plot_score_distributions(
    member_scores: np.ndarray,
    non_member_scores: np.ndarray,
    score_type: str = "loss",
    fit_normal: bool = True,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot histograms of member vs non-member MIA scores, optionally with fitted Gaussians.
    """
    _default_style()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    bins = np.linspace(
        min(member_scores.min(), non_member_scores.min()),
        max(member_scores.max(), non_member_scores.max()),
        50,
    )
    ax.hist(member_scores, bins=bins, alpha=0.5, density=True, label="Member", color="steelblue", edgecolor="none")
    ax.hist(non_member_scores, bins=bins, alpha=0.5, density=True, label="Non-member", color="coral", edgecolor="none")

    if fit_normal:
        fit_m, fit_n = fit_score_distributions(member_scores, non_member_scores, family="normal")
        x = np.linspace(bins.min(), bins.max(), 200)
        ax.plot(x, fit_m.pdf(x), color="steelblue", lw=2, label="Member (fit)")
        ax.plot(x, fit_n.pdf(x), color="coral", lw=2, label="Non-member (fit)")

    ax.set_xlabel("MIA score" if score_type != "loss" else "Cross-entropy loss")
    ax.set_ylabel("Density")
    ax.set_title("Member vs non-member score distributions")
    ax.legend(loc="upper right")
    ax.set_ylim(0, None)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        (fig or plt.gcf()).savefig(save_path, dpi=150, bbox_inches="tight")
    return fig or plt.gcf()


def plot_roc_curve(
    member_scores: np.ndarray,
    non_member_scores: np.ndarray,
    score_higher_is_non_member: bool = True,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot ROC curve (TPR vs FPR)."""
    _default_style()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    fpr, tpr, _ = roc_curve(
        member_scores,
        non_member_scores,
        score_higher_is_non_member=score_higher_is_non_member,
    )
    ax.plot(fpr, tpr, color="darkgreen", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (MIA)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        (fig or plt.gcf()).savefig(save_path, dpi=150, bbox_inches="tight")
    return fig or plt.gcf()


def plot_precision_recall(
    member_scores: np.ndarray,
    non_member_scores: np.ndarray,
    score_higher_is_non_member: bool = True,
    n_points: int = 101,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot precision-recall curve."""
    _default_style()
    from sklearn.metrics import precision_recall_curve, auc

    y_true = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])
    y_score = np.concatenate([member_scores, non_member_scores])
    if score_higher_is_non_member:
        y_score = -y_score
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = auc(rec, prec)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(rec, prec, color="darkviolet", lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curve (MIA)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        (fig or plt.gcf()).savefig(save_path, dpi=150, bbox_inches="tight")
    return fig or plt.gcf()


def plot_neyman_pearson_curve(
    member_scores: np.ndarray,
    non_member_scores: np.ndarray,
    score_higher_is_non_member: bool = True,
    alphas: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot power vs alpha (FPR) for Neyman–Pearson test.
    """
    _default_style()
    if alphas is None:
        alphas = np.linspace(0.01, 0.5, 30)

    results = [
        neyman_pearson_test(
            member_scores,
            non_member_scores,
            alpha=float(a),
            score_higher_is_non_member=score_higher_is_non_member,
        )
        for a in alphas
    ]
    fprs = [r["fpr"] for r in results]
    powers = [r["power"] for r in results]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(alphas, powers, "o-", color="teal", lw=2, markersize=4, label="Power")
    ax.plot(alphas, fprs, "s--", color="gray", lw=1, markersize=3, label="Empirical FPR")
    ax.set_xlabel(r"Target $\alpha$ (type-I error)")
    ax.set_ylabel("Power / FPR")
    ax.set_title("Neyman–Pearson: Power vs target FPR")
    ax.legend(loc="lower right")
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        (fig or plt.gcf()).savefig(save_path, dpi=150, bbox_inches="tight")
    return fig or plt.gcf()
