"""Neyman–Pearson test for membership inference."""

from typing import Optional, Tuple

import numpy as np
from scipy import stats

from .distributions import FittedDistribution, fit_score_distributions


def optimal_threshold(
    member_scores: np.ndarray,
    non_member_scores: np.ndarray,
    alpha: float,
    score_higher_is_non_member: bool = True,
    use_empirical: bool = True,
) -> Tuple[float, float, float]:
    """
    Compute decision threshold for target type-I error (false positive rate).

    We fix P(say member | non-member) = alpha and choose threshold.
    - If score_higher_is_non_member: reject (say member) when score < threshold.
    - Otherwise: reject when score > threshold.

    Args:
        member_scores: MIA scores for members.
        non_member_scores: MIA scores for non-members.
        alpha: Target type-I error (FPR).
        score_higher_is_non_member: True for loss-based scores.
        use_empirical: Use empirical quantiles; else fit Gaussians.

    Returns:
        (threshold, empirical_power, empirical_fpr)
    """
    if use_empirical:
        if score_higher_is_non_member:
            # Say "member" when score < t. FPR = P(score < t | non-member) = alpha => t = quantile(non_member, alpha)
            # But we want FPR = alpha, so we use lower tail of non_member.
            threshold = np.quantile(non_member_scores, alpha)
        else:
            # Say "member" when score > t. FPR = P(score > t | non-member) = alpha => t = quantile(non_member, 1-alpha)
            threshold = np.quantile(non_member_scores, 1 - alpha)

        pred_member = member_scores < threshold if score_higher_is_non_member else member_scores > threshold
        pred_non = non_member_scores < threshold if score_higher_is_non_member else non_member_scores > threshold
        power = np.mean(pred_member)
        fpr = np.mean(pred_non)
        return float(threshold), float(power), float(fpr)

    fit_m, fit_n = fit_score_distributions(member_scores, non_member_scores, family="normal")
    if score_higher_is_non_member:
        threshold = float(np.clip(fit_n.ppf(alpha), -1e10, 1e10))
        power = float(np.mean(member_scores < threshold))
        fpr = float(np.mean(non_member_scores < threshold))
    else:
        threshold = float(np.clip(fit_n.ppf(1 - alpha), -1e10, 1e10))
        power = float(np.mean(member_scores > threshold))
        fpr = float(np.mean(non_member_scores > threshold))
    return threshold, power, fpr


def neyman_pearson_test(
    member_scores: np.ndarray,
    non_member_scores: np.ndarray,
    alpha: float = 0.05,
    score_higher_is_non_member: bool = True,
    fit_family: str = "normal",
) -> dict:
    """
    Neyman–Pearson style test: fix FPR at alpha, report threshold and power.

    Fits member and non-member score distributions, then computes the
    threshold that achieves target type-I error and the resulting power.

    Args:
        member_scores: MIA scores for members.
        non_member_scores: MIA scores for non-members.
        alpha: Target type-I error (FPR).
        score_higher_is_non_member: True for loss (higher = non-member).
        fit_family: 'normal' for Gaussian fitting (kernel not supported for NP test).

    Returns:
        Dict with threshold, power, fpr, and fitted params.
    """
    if fit_family != "normal":
        raise NotImplementedError("Neyman-Pearson test only supports fit_family='normal'.")
    fit_m, fit_n = fit_score_distributions(member_scores, non_member_scores, family=fit_family)

    if score_higher_is_non_member:
        # Reject H0 (non-member) when score < t. FPR = P(score < t | non-member) = F_n(t) = alpha => t = F_n^{-1}(alpha)
        try:
            threshold = float(fit_n.ppf(np.clip(alpha, 1e-6, 1 - 1e-6)))
        except Exception:
            threshold = np.quantile(non_member_scores, alpha)
    else:
        try:
            threshold = float(fit_n.ppf(np.clip(1 - alpha, 1e-6, 1 - 1e-6)))
        except Exception:
            threshold = np.quantile(non_member_scores, 1 - alpha)

    # Empirical power and FPR at this threshold
    if score_higher_is_non_member:
        say_member = member_scores < threshold
        say_member_n = non_member_scores < threshold
    else:
        say_member = member_scores > threshold
        say_member_n = non_member_scores > threshold

    power = float(np.mean(say_member))
    fpr = float(np.mean(say_member_n))

    return {
        "threshold": threshold,
        "power": power,
        "fpr": fpr,
        "alpha": alpha,
        "score_higher_is_non_member": score_higher_is_non_member,
        "member_dist_params": getattr(fit_m, "params", None),
        "non_member_dist_params": getattr(fit_n, "params", None),
    }


def roc_curve(
    member_scores: np.ndarray,
    non_member_scores: np.ndarray,
    score_higher_is_non_member: bool = True,
    n_points: int = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve (FPR, TPR) and thresholds.

    Returns:
        (fpr, tpr, thresholds)
    """
    if score_higher_is_non_member:
        # "Positive" = member. Predict member when score < t.
        all_scores = np.concatenate([member_scores, non_member_scores])
        thresh = np.percentile(all_scores, np.linspace(0, 100, n_points))
        thresh = np.unique(thresh)
        tpr = np.array([np.mean(member_scores < t) for t in thresh])
        fpr = np.array([np.mean(non_member_scores < t) for t in thresh])
    else:
        all_scores = np.concatenate([member_scores, non_member_scores])
        thresh = np.percentile(all_scores, np.linspace(0, 100, n_points))
        thresh = np.unique(thresh)
        tpr = np.array([np.mean(member_scores > t) for t in thresh])
        fpr = np.array([np.mean(non_member_scores > t) for t in thresh])
    return fpr, tpr, thresh
