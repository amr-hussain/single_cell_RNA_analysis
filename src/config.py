from pathlib import Path
import yaml


def load_config(path):
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['root'] = Path(path).resolve().parent
    return cfg
