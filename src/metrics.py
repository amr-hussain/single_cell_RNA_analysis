import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

METRICS = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'kappa', 'auc_roc']


def classification_metrics(y_true, pred, proba, classes):
    """Tabular-data metrics. Macro = every cell type counts equally (rare types matter)."""
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, pred, labels=classes, average='macro', zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, proba, multi_class='ovr', average='macro', labels=classes)
    except ValueError:
        auc = np.nan    # a class missing from this test fold
    return dict(
        accuracy=accuracy_score(y_true, pred),
        macro_precision=prec,
        macro_recall=rec,
        macro_f1=f1,
        kappa=cohen_kappa_score(y_true, pred),
        auc_roc=auc,
    )


# ── Embedding quality: does the low-dim picture respect the original geometry? ──

def continuity(X, Z, k=10):
    """Continuity = trustworthiness with the roles of the two spaces swapped.

    Trustworthiness penalises *false* neighbours that appear in the embedding;
    continuity penalises *true* neighbours that the embedding tore apart.
    """
    return trustworthiness(Z, X, n_neighbors=k)


def knn_preservation(X, Z, k=10):
    """Average fraction of each cell's k nearest neighbours that survive the embedding."""
    a = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    b = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    return float(np.mean([len(set(r) & set(s)) / k for r, s in zip(a, b)]))


def cluster_scores(Z, y, seed):
    """k-means with k = #cell types → ARI vs true labels, and silhouette of the true labels."""
    k = len(np.unique(y))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    return dict(
        ari=adjusted_rand_score(y, km.labels_),
        silhouette=silhouette_score(Z, y) if k > 1 else np.nan,
    )


def embedding_quality(X, Z, y, k, seed, max_cells, rng=None):
    """All embedding-quality metrics on a (sub)sample of cells; X is the reference space."""
    if len(X) > max_cells:
        rng = rng or np.random.default_rng(seed)
        ids = rng.choice(len(X), max_cells, replace=False)
        X, Z, y = X[ids], Z[ids], y[ids]
    out = dict(
        trustworthiness=trustworthiness(X, Z, n_neighbors=k),
        continuity=continuity(X, Z, k),
        knn_preservation=knn_preservation(X, Z, k),
    )
    out.update(cluster_scores(Z, y, seed))
    return out
