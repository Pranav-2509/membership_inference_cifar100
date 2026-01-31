"""Distribution fitting for MIA score distributions."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class FittedDistribution:
    """Fitted distribution (e.g. normal) with params."""

    name: str
    params: dict
    dist: stats.rv_continuous

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self.dist.pdf(x, **self.params)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self.dist.cdf(x, **self.params)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        return self.dist.ppf(q, **self.params)


def fit_normal(scores: np.ndarray) -> FittedDistribution:
    """Fit Gaussian to scores. Returns mean and std."""
    mu, std = np.mean(scores), np.std(scores)
    if std <= 0:
        std = 1e-8
    return FittedDistribution(
        name="normal",
        params={"loc": mu, "scale": std},
        dist=stats.norm,
    )


def fit_score_distributions(
    member_scores: np.ndarray,
    non_member_scores: np.ndarray,
    family: str = "normal",
) -> tuple[FittedDistribution, FittedDistribution]:
    """
    Fit a parametric distribution to member and non-member score samples.

    Args:
        member_scores: MIA scores for members.
        non_member_scores: MIA scores for non-members.
        family: 'normal' (Gaussian) or 'kernel' (KDE, stored differently).

    Returns:
        (fitted_member, fitted_non_member)
    """
    if family == "normal":
        return fit_normal(member_scores), fit_normal(non_member_scores)
    if family == "kernel":
        fm, fn = fit_kde(member_scores), fit_kde(non_member_scores)
        return fm, fn
    raise ValueError(f"Unknown family: {family}")


def fit_kde(scores: np.ndarray) -> "FittedKDE":
    """Fit kernel density estimate."""
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(scores.T if scores.ndim > 1 else scores)
    return FittedKDE(kde=kde, scores=scores)


class FittedKDE:
    """Thin wrapper around scipy KDE for unified interface."""

    def __init__(self, kde: "gaussian_kde", scores: np.ndarray) -> None:
        self.kde = kde
        self.scores = np.atleast_1d(scores)
        if self.scores.ndim == 1:
            self.scores = self.scores.reshape(-1, 1)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return self.kde.pdf(x.T).ravel()

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Approximate CDF via grid integration (avoids quad over (-inf, xi))."""
        x = np.atleast_1d(x).astype(float)
        out = np.empty_like(x)
        lo = float(self.scores.min() - 5 * np.std(self.scores))
        for i, xi in enumerate(x.ravel()):
            grid = np.linspace(lo, xi, 200)
            if grid.size:
                g = grid.reshape(1, -1) if grid.ndim == 1 else grid
                probs = self.kde.pdf(g).ravel()
                out.flat[i] = np.trapz(probs, grid)
            else:
                out.flat[i] = 0.0
        return out


def fit_normal_mixture(
    scores: np.ndarray,
    n_components: int = 2,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit Gaussian mixture to scores.

    Returns:
        (weights, means, stds) of shape (n_components,).
    """
    from sklearn.mixture import GaussianMixture

    X = np.atleast_1d(scores).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=random_state, covariance_type="diag")
    gmm.fit(X)
    weights = gmm.weights_
    means = gmm.means_.ravel()
    stds = np.sqrt(gmm.covariances_.ravel())
    return weights, means, stds
