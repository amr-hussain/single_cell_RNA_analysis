"""Run the whole tutorial pipeline from the command line.

    python run_pipeline.py              # full configuration
    python run_pipeline.py --fast       # quick check: 800 cells, 1 repeat, fewer dims/k
    python run_pipeline.py --parts A,C  # only some parts (baseline always runs if A/B need it)
"""
import argparse
import logging
import time

import matplotlib
matplotlib.use('Agg')   # no windows: figures are only saved to files (the notebook keeps its inline backend)

from src.config import load_config, set_seed
from src.cv import make_splits
from src.data import load_dataset
from src.experiments import run_baseline, run_data_summary, run_partA, run_partB, run_partC

log = logging.getLogger('tutorial')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--fast', action='store_true', help='apply the `fast:` block of the config')
    ap.add_argument('--parts', default='baseline,A,B,C', help='comma list from: baseline,A,B,C')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    cfg = load_config(args.config, fast=args.fast)
    set_seed(cfg['seed'])
    parts = {p.strip() for p in args.parts.split(',')}
    t0 = time.perf_counter()

    log.info('Mode: %s · dataset: %s · features: %s', 'FAST' if args.fast else 'FULL', cfg['dataset'], cfg['feature_source'])
    ds = load_dataset(cfg)
    run_data_summary(ds, cfg)
    splits = make_splits(ds.y, cfg)

    base = None
    if parts & {'baseline', 'A', 'B'}:
        base = run_baseline(ds, cfg, splits)
        log.info('\n%s', base['summary'][['classifier', 'accuracy_mean', 'macro_f1_mean', 'total_cpu_mean']].to_string(index=False))
    if 'A' in parts:
        run_partA(ds, cfg, base, splits)
    if 'B' in parts:
        run_partB(ds, cfg, base, splits)
    if 'C' in parts:
        run_partC(ds, cfg)

    log.info('All done in %.1f min. Tables: %s  Figures: %s',
             (time.perf_counter() - t0) / 60, cfg['tables_dir'], cfg['figures_dir'])


if __name__ == '__main__':
    main()
