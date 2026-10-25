"""Dimensionality reducers behind one small interface.

Each reducer is a sklearn Pipeline: scale+clip → (PCA-50 pre-step) → method.
- Inductive methods (PCA, Kernel PCA, Isomap, UMAP) learn a mapping: fit on the
  training fold, then `transform` the test fold. No leakage.
- Transductive methods (t-SNE, MDS, SNE) only place the points they were given,
  so train + test are embedded together. They are flagged everywhere.
"""
import inspect
import time
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.pipeline import Pipeline
from .data import make_scaler
from .sne import GaussianSNE

INDUCTIVE = {'PCA', 'Kernel PCA', 'Isomap', 'UMAP'}
TRANSDUCTIVE = {'SNE', 't-SNE', 'MDS'}
NEEDS_PCA_PRE = {'SNE', 't-SNE', 'MDS', 'Isomap', 'UMAP'}


class SkipMethod(Exception):
    """Raised when a method/dimension combination is deliberately not run."""


def accepts(cls, param):
    """True if this scikit-learn version's estimator has the given constructor argument."""
    return param in inspect.signature(cls.__init__).parameters


def timed(fn, *args):
    """Run fn(*args) and return (result, cpu_seconds, wall_seconds).

    process_time counts CPU used by *all* threads of this process, so with
    multithreaded code CPU time can exceed wall-clock time.
    """
    t0 = time.process_time()
    w0 = time.perf_counter()
    out = fn(*args)
    return out, time.process_time() - t0, time.perf_counter() - w0


class SNEWrapper(BaseEstimator, TransformerMixin):
    """Lets the custom GaussianSNE sit inside a Pipeline (fit_transform only)."""

    def __init__(self, n_components=2, perplexity=30, n_iter=500, learning_rate='auto', random_state=42):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        self.model_ = GaussianSNE(self.n_components, self.perplexity, self.n_iter,
                                  self.learning_rate, self.random_state)
        return self.model_.fit_transform(X)

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self


def _method(name, dim, cfg, seed, n_samples, n_features):
    h = cfg['hyper']
    a = cfg['partA']
    if name == 'PCA':
        return PCA(n_components=dim, random_state=seed)
    if name == 'Kernel PCA':
        return KernelPCA(n_components=dim, kernel=h['kpca_kernel'], gamma=h['kpca_gamma'],
                         eigen_solver='randomized', random_state=seed)
    if name == 't-SNE':
        method = 'barnes_hut'
        if dim > 3:
            if a['tsne_high_dim'] == 'skip':
                raise SkipMethod(f't-SNE: Barnes-Hut supports ≤3 dims; dim={dim} skipped (tsne_high_dim=skip)')
            method = 'exact'
        # perplexity must be < n_samples
        perp = min(h['tsne_perplexity'], (n_samples - 1) / 3)
        iters = 'max_iter' if accepts(TSNE, 'max_iter') else 'n_iter'   # renamed in scikit-learn 1.5
        return TSNE(n_components=dim, perplexity=perp, method=method, init='pca', learning_rate='auto',
                    random_state=seed, **{iters: h['tsne_iterations']})
    if name == 'SNE':
        if dim > a['sne_max_dim']:
            raise SkipMethod(f'SNE: dim={dim} > sne_max_dim={a["sne_max_dim"]}, skipped')
        perp = min(h['sne_perplexity'], (n_samples - 1) / 3)
        return SNEWrapper(dim, perp, h['sne_iterations'], h['sne_learning_rate'], seed)
    if name == 'MDS':
        kw = {}
        if accepts(MDS, 'init'):      # scikit-learn ≥ 1.8: start from classical MDS instead of random
            kw['init'] = 'classical_mds'
        return MDS(n_components=dim, n_init=h['mds_n_init'], max_iter=h['mds_max_iter'],
                   random_state=seed, normalized_stress='auto', **kw)
    if name == 'Isomap':
        return Isomap(n_components=dim, n_neighbors=h['isomap_neighbors'])
    if name == 'UMAP':
        import umap
        return umap.UMAP(n_components=dim, n_neighbors=h['umap_neighbors'], min_dist=h['umap_min_dist'],
                         random_state=seed)
    raise ValueError(f'Unknown reducer {name}')


def make_reducer(name, dim, cfg, seed, n_samples, n_features):
    """Build Pipeline(scale → [PCA-50] → method). Raises SkipMethod if not applicable."""
    pre = min(cfg['partA']['pca_pre_dims'], n_features, n_samples - 1)
    if dim > min(n_features, n_samples - 1):
        raise SkipMethod(f'{name}: dim={dim} exceeds data rank')
    if name in NEEDS_PCA_PRE and dim > pre:
        raise SkipMethod(f'{name}: dim={dim} > PCA pre-step ({pre}), skipped')
    steps = [('scale', make_scaler(cfg))]
    if name in NEEDS_PCA_PRE and pre < n_features:
        steps.append(('pca_pre', PCA(n_components=pre, random_state=seed)))
    steps.append(('reduce', _method(name, dim, cfg, seed, n_samples, n_features)))
    return Pipeline(steps)


def is_inductive(name):
    return name in INDUCTIVE
