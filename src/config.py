import os
import random
from pathlib import Path
import numpy as np
import yaml


def _merge(base, extra):
    """Recursively overlay `extra` on `base` (used for the `fast:` block)."""
    for key, val in extra.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _merge(base[key], val)
        else:
            base[key] = val
    return base


def load_config(path='config.yaml', fast=False):
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    fast_block = cfg.pop('fast', {}) or {}
    if fast:
        _merge(cfg, fast_block)
    cfg['fast_mode'] = fast
    cfg['root'] = Path(path).resolve().parent
    for key in ('tables', 'figures'):
        p = cfg['root'] / cfg['outputs'][key].format(mode='fast' if fast else 'full')
        p.mkdir(parents=True, exist_ok=True)
        cfg[f'{key}_dir'] = p
    return cfg


def set_seed(seed):
    """Seed every global random source we use (estimators also get random_state)."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
