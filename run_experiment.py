import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import load_config
from src.data import benchmark_sample
from src.reducers import run_reducers
from src.metrics import evaluate, froc_curve
from src.plots import save_embedding, save_cm, save_roc, save_froc, comparison
from src.statistics import compare_methods, manova_text
from src.report import build_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg['root']

    fig = root / 'outputs/figures'
    tab = root / 'outputs/tables'
    rep = root / 'report'
    for p in [fig, tab, rep]:
        p.mkdir(parents=True, exist_ok=True)

    repeats = []
    perclass = []
    first_payload = {}

    for repeat in range(cfg['repeats']):
        seed = cfg['random_seed'] + repeat
        X, y, mask = benchmark_sample(
            root / 'data', cfg['train_per_class'], cfg['test_per_class'], seed
        )
        for r in run_reducers(X, cfg, seed):
            if 'subset_index' in r:
                ids = r['subset_index']
                Z = r['embedding']
                yy = y[ids]
                mm = mask[ids]
                # Ensure both partitions and every class remain represented.
                if mm.sum() < 20 or (~mm).sum() < 20:
                    continue
            else:
                Z = r['embedding']
                yy = y
                mm = mask

            s, pc, cm, yt, pred, proba, _ = evaluate(
                r['method'], Z, yy, mm, cfg['classifier_neighbors']
            )
            xfr, yfr, af = froc_curve(yt, proba, cfg['froc_thresholds'])
            s.update(
                cpu_seconds=r['cpu_seconds'],
                wall_seconds=r['wall_seconds'],
                froc_auc=af,
                repeat=repeat,
            )
            repeats.append(s)
            pc['repeat'] = repeat
            perclass.append(pc)
            if repeat == 0:
                first_payload[r['method']] = (Z, yy, cm, yt, proba, xfr, yfr, af)

    df = pd.DataFrame(repeats)
    pcdf = pd.concat(perclass, ignore_index=True)
    metrics = df.groupby('method', as_index=False).mean(numeric_only=True)
    tests = compare_methods(df)
    manova = manova_text(df)

    df.to_csv(tab / 'repeat_metrics.csv', index=False)
    metrics.to_csv(tab / 'metrics.csv', index=False)
    pcdf.to_csv(tab / 'per_class_metrics.csv', index=False)
    tests.to_csv(tab / 'statistical_tests.csv', index=False)
    (tab / 'manova.txt').write_text(manova, encoding='utf-8')

    for name, (Z, yy, cm, yt, proba, xfr, yfr, af) in first_payload.items():
        save_embedding(Z, yy, name, fig)
        save_cm(cm, name, fig)
        save_roc(yt, proba, name, fig)
        save_froc(xfr, yfr, af, name, fig)

    comparison(metrics, fig)
    meta = {
        'n': len(X),
        'n_train': int(mask.sum()),
        'n_test': int((~mask).sum()),
    }
    build_report(metrics, None, tests, manova, fig, rep / 'Fashion_MNIST_DR_Report.docx', meta)
    print(metrics.to_string(index=False))
    print('\nReport:', rep / 'Fashion_MNIST_DR_Report.docx')


if __name__ == '__main__':
    main()
