from pathlib import Path
import os
import gzip
import struct
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def _idx_images(path):
    with gzip.open(path, 'rb') as f:
        magic, n, r, c = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f'Bad image IDX magic in {path}')
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c)


def _idx_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f'Bad label IDX magic in {path}')
        return np.frombuffer(f.read(), dtype=np.uint8)


def _find(root, names):
    for name in names:
        hits = list(Path(root).rglob(name))
        if hits:
            return hits[0]
    return None


def load_fashion_mnist(root='data'):
    roots = [Path(root), Path(os.getenv('FASHION_MNIST_DIR', ''))]
    for base in roots:
        if not str(base):
            continue
        tr = _find(base, ['fashion-mnist_train.csv', 'fashion-mnist_train.csv.gz'])
        te = _find(base, ['fashion-mnist_test.csv', 'fashion-mnist_test.csv.gz'])
        if tr and te:
            a = pd.read_csv(tr)
            b = pd.read_csv(te)
            return (
                a.iloc[:, 1:].to_numpy(np.float32),
                a.iloc[:, 0].to_numpy(),
                b.iloc[:, 1:].to_numpy(np.float32),
                b.iloc[:, 0].to_numpy(),
            )
        ti = _find(base, ['train-images-idx3-ubyte.gz'])
        tl = _find(base, ['train-labels-idx1-ubyte.gz'])
        vi = _find(base, ['t10k-images-idx3-ubyte.gz'])
        vl = _find(base, ['t10k-labels-idx1-ubyte.gz'])
        if all([ti, tl, vi, vl]):
            return _idx_images(ti), _idx_labels(tl), _idx_images(vi), _idx_labels(vl)
    import kagglehub
    path = kagglehub.dataset_download('zalando-research/fashionmnist')
    return load_fashion_mnist(path)


def balanced_sample(X, y, per_class, seed):
    rng = np.random.default_rng(seed)
    ids = []
    for c in np.unique(y):
        ids.extend(rng.choice(np.flatnonzero(y == c), min(per_class, (y == c).sum()), replace=False))
    ids = np.array(ids)
    rng.shuffle(ids)
    return X[ids].astype(np.float32) / 255.0, y[ids]


def benchmark_sample(root, train_per_class, test_per_class, seed):
    Xtr, ytr, Xte, yte = load_fashion_mnist(root)
    Xtr, ytr = balanced_sample(Xtr, ytr, train_per_class, seed)
    Xte, yte = balanced_sample(Xte, yte, test_per_class, seed + 1)
    X = np.vstack([Xtr, Xte])
    y = np.r_[ytr, yte]
    train_mask = np.arange(len(X)) < len(Xtr)
    return X, y, train_mask
