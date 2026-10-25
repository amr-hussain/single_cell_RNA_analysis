"""Feature (gene) selection: keep k ORIGINAL genes, create no new coordinates.

All selectors are sklearn transformers, so they go inside a Pipeline and are
fit on the training fold only. Gene scores are cached per (method, training
data) so trying k = 10, 25, 50, ... reuses one scoring pass. The recorded
selection time is still the full scoring cost, as if each k were run alone.
"""
import hashlib
import inspect
import time
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def fisher_score(X, y):
    """
    Compute Fisher Score for each feature.

    Fisher Score = Σ_c  n_c * (μ_c - μ)^2 / Σ_c  n_c * σ_c^2

    Higher is better (more discriminative).
    """
    classes = np.unique(y)
    n_features = X.shape[1]
    overall_mean = X.mean(axis=0)
    numerator = np.zeros(n_features)
    denominator = np.zeros(n_features)

    for c in classes:
        Xc = X[y == c]
        nc = len(Xc)
        class_mean = Xc.mean(axis=0)
        class_var = Xc.var(axis=0)
        numerator += nc * (class_mean - overall_mean) ** 2
        denominator += nc * class_var

    denominator[denominator == 0] = 1e-12  # avoid division by zero
    return numerator / denominator


def _variance(X, y, seed, cfg):
    # Unsupervised: ignores labels. Needs UNSCALED data (after z-scoring every variance is 1).
    return X.var(axis=0)


def _anova(X, y, seed, cfg):
    return np.nan_to_num(f_classif(X, y)[0], nan=0.0)


def _mutual_info(X, y, seed, cfg):
    return mutual_info_classif(X, y, random_state=seed, n_jobs=-1)


def _fisher(X, y, seed, cfg):
    return fisher_score(X, y)


def _lasso(X, y, seed, cfg):
    # L1-penalised multinomial logistic regression on scaled genes; a gene's score is
    # its largest |coefficient| over classes. Most genes get exactly 0.
    Xs = StandardScaler().fit_transform(X)
    # scikit-learn ≥ 1.8 deprecates `penalty` (L1 = l1_ratio=1); older versions need penalty='l1'.
    if inspect.signature(LogisticRegression).parameters['penalty'].default == 'deprecated':
        l1 = dict(l1_ratio=1.0)
    else:
        l1 = dict(penalty='l1')
    lr = LogisticRegression(solver='saga', C=cfg.get('lasso_C', 0.1), max_iter=300, tol=1e-3,
                            random_state=seed, **l1)
    lr.fit(Xs, y)
    return np.abs(lr.coef_).max(axis=0)


SCORERS = {
    'Variance Threshold': _variance,
    'ANOVA F-test': _anova,
    'Mutual Information': _mutual_info,
    'Fisher Score': _fisher,
    'Lasso (L1)': _lasso,
}

_CACHE = {}


def gene_scores(method, X, y, seed=42, cfg=None):
    """Return (scores, cpu_seconds, wall_seconds) — cached on the exact training data."""
    key = (method, seed, hashlib.blake2b(X.tobytes() + y.tobytes(), digest_size=16).hexdigest())
    if key not in _CACHE:
        t0, w0 = time.process_time(), time.perf_counter()
        scores = SCORERS[method](X, y, seed, cfg or {})
        _CACHE[key] = (scores, time.process_time() - t0, time.perf_counter() - w0)
    return _CACHE[key]


def top_k(scores, k):
    """Indices of the k highest scores (ties broken by gene order → deterministic)."""
    return np.argsort(-scores, kind='stable')[:k]


class TopKSelector(BaseEstimator, TransformerMixin):
    """Keep the k genes with the highest score under `method` (fit on training data only)."""

    def __init__(self, method='ANOVA F-test', k=50, seed=42, cfg=None):
        self.method = method
        self.k = k
        self.seed = seed
        self.cfg = cfg

    def fit(self, X, y):
        self.scores_, self.cpu_seconds_, self.wall_seconds_ = gene_scores(self.method, X, y, self.seed, self.cfg)
        self.support_ = top_k(self.scores_, min(self.k, X.shape[1]))
        return self

    def transform(self, X):
        return X[:, self.support_]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)
