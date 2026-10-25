"""Dataset loaders.

Every loader returns an AnnData plus the name of the label column. `load_dataset`
then turns it into plain numpy arrays. To add a dataset, write a loader and
register it in LOADERS; then change `dataset:` in config.yaml.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def _dense(M):
    return np.asarray(M.toarray() if sp.issparse(M) else M, dtype=np.float32)


def _pbmc3k(sc):
    # 2638 PBMCs, already filtered/normalised/log1p/scaled; .raw = log-normalised.
    return sc.datasets.pbmc3k_processed(), 'louvain'


def _pbmc68k(sc):
    # 700-cell subsample of 68k PBMCs, labels from FACS-sorted bulk populations.
    return sc.datasets.pbmc68k_reduced(), 'bulk_labels'


def _paul15(sc):
    # Mouse myeloid progenitors: raw counts → we do the standard preprocessing here.
    adata = sc.datasets.paul15()
    adata.raw = None
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    adata = adata[:, adata.var['highly_variable']].copy()
    return adata, 'paul15_clusters'


LOADERS = {
    'pbmc3k_processed': _pbmc3k,
    'pbmc68k_reduced': _pbmc68k,
    'paul15': _paul15,
}


@dataclass
class Dataset:
    X: np.ndarray            # cells × genes, float32
    y: np.ndarray            # integer-encoded labels
    class_names: list
    gene_names: np.ndarray
    name: str
    feature_source: str


def stratified_subsample(y, n, min_per_class, seed):
    """Indices of a stratified subsample of ~n cells, keeping rare classes ≥ min_per_class."""
    rng = np.random.default_rng(seed)
    ids = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        take = max(int(round(n * len(members) / len(y))), min_per_class)
        ids.extend(rng.choice(members, min(take, len(members)), replace=False))
    return np.sort(np.array(ids))


def load_dataset(cfg):
    import scanpy as sc
    sc.settings.datasetdir = Path(cfg['root']) / cfg.get('data_dir', 'data')
    sc.settings.verbosity = 0

    name = cfg['dataset']
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(LOADERS)}")
    adata, label_key = LOADERS[name](sc)

    source = cfg.get('feature_source', 'lognorm_hvg')
    if source == 'lognorm_hvg' and adata.raw is not None:
        X = _dense(adata.raw[:, adata.var_names].X)
    else:
        X = _dense(adata.X)

    labels = pd.Categorical(adata.obs[label_key])
    y = labels.codes.astype(int)
    genes = np.asarray(adata.var_names)

    if cfg.get('subsample_cells'):
        ids = stratified_subsample(y, cfg['subsample_cells'], cfg.get('min_cells_per_class', 10), cfg['seed'])
        X, y = X[ids], y[ids]

    # Genes that are constant in the (sub)sample carry no information and break F-tests.
    keep = X.std(axis=0) > 0
    return Dataset(X[:, keep], y, list(labels.categories), genes[keep], name, source)


def make_scaler(cfg):
    """z-score each gene, then clip at ±clip_value (as scanpy's sc.pp.scale(max_value=10)).

    Without clipping, a gene expressed in only a few cells gets huge z-scores and
    those few cells dominate distances and variance.
    """
    clip = cfg.get('clip_value')
    steps = [('zscore', StandardScaler())]
    if clip:
        steps.append(('clip', FunctionTransformer(np.clip, kw_args=dict(a_min=-clip, a_max=clip))))
    return Pipeline(steps)


def summarize(ds):
    counts = pd.Series(np.bincount(ds.y, minlength=len(ds.class_names)), index=ds.class_names)
    counts = counts.sort_values(ascending=False)
    ratio = counts.max() / max(counts.min(), 1)
    return {
        'cells': len(ds.y),
        'genes': ds.X.shape[1],
        'classes': len(ds.class_names),
        'counts': counts,
        'imbalance_ratio': ratio,
        'note': (
            f'Largest class is {ratio:.1f}× the smallest. '
            + ('This is strongly imbalanced, so macro-averaged metrics (every class '
               'weighs the same) are more honest than plain accuracy.' if ratio > 5 else
               'Moderate imbalance; still report macro metrics alongside accuracy.')
        ),
    }
