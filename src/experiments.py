"""The four experiments. Each writes its tables/figures and returns a dict of results,
so `run_pipeline.py` and the tutorial notebook share exactly the same code."""
import logging
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from .data import make_scaler

from . import plots
from .cv import eval_classifiers, fit_predict, make_classifiers, make_splits
from .data import summarize
from .metrics import METRICS, classification_metrics, embedding_quality
from .reducers import SkipMethod, is_inductive, make_reducer, timed
from .selection import TopKSelector, gene_scores, jaccard, top_k
from .statistics import anova_across_methods, compare_to_baseline

log = logging.getLogger('tutorial')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*n_jobs value.*')          # UMAP + random_state
warnings.filterwarnings('ignore', message='.*n_quantiles.*')
# Some genes are all-zero inside a small training fold → ANOVA divides 0/0; we map NaN → 0.
warnings.filterwarnings('ignore', message='Features .* are constant')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
# Lasso (saga) is capped at max_iter for speed; the ranking of genes is stable well before full convergence.
warnings.filterwarnings('ignore', category=ConvergenceWarning)

def immune_markers(cfg):
    """Flat set of the canonical marker genes listed per cell type in config.yaml."""
    return {g for genes in cfg.get('markers', {}).values() for g in genes}


def _table(df, cfg, name):
    df.to_csv(cfg['tables_dir'] / name, index=False)
    log.info('  table  → %s', cfg['tables_dir'] / name)


def _fig(path):
    log.info('  figure → %s', path)


def _try_plot(fn, *args, **kw):
    try:
        _fig(fn(*args, **kw))
    except Exception as e:
        log.warning('  plot %s failed: %s', getattr(fn, '__name__', fn), e)


# ── Data summary ─────────────────────────────────────────────────────────────

def run_data_summary(ds, cfg):
    s = summarize(ds)
    log.info('Data: %d cells × %d genes, %d cell types. %s', s['cells'], s['genes'], s['classes'], s['note'])
    t = s['counts'].rename_axis('cell_type').reset_index(name='cells')
    t['fraction'] = t.cells / t.cells.sum()
    _table(t, cfg, 'data_summary.csv')
    _try_plot(plots.class_counts, s, cfg['figures_dir'], cfg['dpi'])
    return s


# ── Baseline ─────────────────────────────────────────────────────────────────

def _time_cols(df):
    df['total_cpu'] = df.filter(regex='(reducer|prep|select).*_cpu$').sum(axis=1) + df.train_cpu + df.pred_cpu
    df['total_wall'] = df.filter(regex='(reducer|prep|select).*_wall$').sum(axis=1) + df.train_wall + df.pred_wall
    return df


def run_baseline(ds, cfg, splits=None):
    """k-NN, SVM and RF on ALL genes (scaler fit on the training fold only)."""
    splits = splits or make_splits(ds.y, cfg)
    classes = np.arange(len(ds.class_names))
    rows = []
    log.info('Baseline: %d folds × 3 classifiers on %d genes', len(splits), ds.X.shape[1])
    for rep, fold, tr, te in splits:
        sc, cpu, wall = timed(make_scaler(cfg).fit, ds.X[tr])
        Xtr, Xte = sc.transform(ds.X[tr]), sc.transform(ds.X[te])
        for r in eval_classifiers(Xtr, ds.y[tr], Xte, ds.y[te], cfg, cfg['seed'] + rep, classes):
            rows.append(dict(method='Baseline', repeat=rep, fold=fold, prep_cpu=cpu, prep_wall=wall, **r))
        log.info('  repeat %d fold %d done', rep, fold)
    df = _time_cols(pd.DataFrame(rows))
    summ = df.groupby('classifier', sort=False)[METRICS + ['train_cpu', 'pred_cpu', 'total_cpu', 'train_wall', 'pred_wall', 'total_wall']].agg(['mean', 'std'])
    summ.columns = [f'{a}_{b}' for a, b in summ.columns]
    summ = summ.reset_index()
    _table(df, cfg, 'baseline_results_folds.csv')
    _table(summ, cfg, 'baseline_summary.csv')
    _try_plot(plots.baseline_bars, df, cfg['figures_dir'], cfg['dpi'])
    return dict(folds=df, summary=summ)


# ── Part A: dimensionality reduction ─────────────────────────────────────────

def run_partA(ds, cfg, base, splits=None):
    splits = splits or make_splits(ds.y, cfg)
    X, y = ds.X, ds.y
    classes = np.arange(len(ds.class_names))
    Xref = make_scaler(cfg).fit_transform(X)       # reference space for quality metrics only
    a = cfg['partA']
    rows, qrows, issues = [], [], []

    for method in a['methods']:
        for dim in a['dims']:
            mode = 'inductive' if is_inductive(method) else 'transductive'
            log.info('Part A: %s → %d dims (%s)', method, dim, mode)
            t_start = time.perf_counter()
            try:
                if is_inductive(method):
                    # Fit on the training fold, transform the test fold: no leakage.
                    for rep, fold, tr, te in splits:
                        seed = cfg['seed'] + rep
                        red = make_reducer(method, dim, cfg, seed, len(tr), X.shape[1])
                        Ztr, fcpu, fwall = timed(red.fit_transform, X[tr])
                        Zte, tcpu, twall = timed(red.transform, X[te])
                        for r in eval_classifiers(Ztr, y[tr], Zte, y[te], cfg, seed, classes):
                            rows.append(dict(method=method, dim=dim, transductive=False, repeat=rep, fold=fold,
                                             reducer_fit_cpu=fcpu, reducer_fit_wall=fwall,
                                             reducer_transform_cpu=tcpu, reducer_transform_wall=twall, **r))
                        if fold == 0:
                            qrows.append(dict(method=method, dim=dim, transductive=False, repeat=rep,
                                              **embedding_quality(Xref[te], Zte, y[te], a['quality_k'], seed, a['quality_max_cells'])))
                else:
                    # Transductive: embed ALL cells once per repeat. Within a repeat, every fold's
                    # train+test = all cells, so this equals per-fold embedding at 1/5 the cost.
                    for rep in range(cfg['cv']['repeats']):
                        seed = cfg['seed'] + rep
                        red = make_reducer(method, dim, cfg, seed, len(X), X.shape[1])
                        Z, fcpu, fwall = timed(red.fit_transform, X)
                        for r_, fold, tr, te in splits:
                            if r_ != rep:
                                continue
                            for r in eval_classifiers(Z[tr], y[tr], Z[te], y[te], cfg, seed, classes):
                                rows.append(dict(method=method, dim=dim, transductive=True, repeat=rep, fold=fold,
                                                 reducer_fit_cpu=fcpu, reducer_fit_wall=fwall,
                                                 reducer_transform_cpu=0.0, reducer_transform_wall=0.0, **r))
                            if fold == 0:
                                qrows.append(dict(method=method, dim=dim, transductive=True, repeat=rep,
                                                  **embedding_quality(Xref[te], Z[te], y[te], a['quality_k'], seed, a['quality_max_cells'])))
                log.info('  done in %.1fs', time.perf_counter() - t_start)
            except SkipMethod as e:
                log.info('  SKIPPED: %s', e)
                issues.append(dict(method=method, dim=dim, status='skipped', reason=str(e)))
            except Exception as e:
                log.warning('  FAILED: %s: %s', type(e).__name__, e)
                issues.append(dict(method=method, dim=dim, status='failed', reason=f'{type(e).__name__}: {e}'))

    df = _time_cols(pd.DataFrame(rows))
    df['reducer_cpu'] = df.reducer_fit_cpu + df.reducer_transform_cpu
    q = pd.DataFrame(qrows)
    bsum = base['folds'].groupby('classifier')[['accuracy', 'macro_f1', 'total_cpu']].mean()

    summ = df.groupby(['method', 'dim', 'classifier', 'transductive'], sort=False)[
        METRICS + ['reducer_fit_cpu', 'reducer_transform_cpu', 'reducer_cpu', 'train_cpu', 'pred_cpu', 'total_cpu', 'total_wall']
    ].mean().reset_index()
    summ['delta_accuracy'] = summ.accuracy - summ.classifier.map(bsum.accuracy)
    summ['delta_macro_f1'] = summ.macro_f1 - summ.classifier.map(bsum.macro_f1)
    summ['speedup_cpu'] = summ.classifier.map(bsum.total_cpu) / summ.total_cpu

    stats_df = pd.concat([compare_to_baseline(df, base['folds'], m, ('method', 'dim')) for m in ['accuracy', 'macro_f1']])
    anova = anova_across_methods(df, base['folds'], 'accuracy', ('classifier', 'dim'))

    _table(df, cfg, 'partA_results_folds.csv')
    _table(summ, cfg, 'partA_summary.csv')
    _table(q, cfg, 'partA_embedding_quality.csv')
    _table(stats_df, cfg, 'partA_stats_vs_baseline.csv')
    _table(anova, cfg, 'partA_anova.csv')
    _table(pd.DataFrame(issues, columns=['method', 'dim', 'status', 'reason']), cfg, 'partA_skipped_or_failed.csv')

    methods = [m for m in a['methods'] if m in set(df.method)]
    F, dpi = cfg['figures_dir'], cfg['dpi']
    _try_plot(plots.metric_lines, df, base['folds'], 'dim', 'accuracy', methods, F, 'partA_accuracy_vs_dim.png', dpi,
              'Target dimensions', 'Part A — accuracy after reduction ((T) = transductive, dashed)', logx=True)
    _try_plot(plots.metric_lines, df, base['folds'], 'dim', 'macro_f1', methods, F, 'partA_f1_vs_dim.png', dpi,
              'Target dimensions', 'Part A — macro F1 after reduction ((T) = transductive, dashed)', logx=True)
    _try_plot(plots.cpu_and_speedup, summ, methods, F, dpi)
    _try_plot(plots.delta_heatmap, summ, methods, F, dpi)
    if not q.empty:
        _try_plot(plots.quality_lines, q, methods, F, dpi)
    return dict(folds=df, summary=summ, quality=q, stats=stats_df, anova=anova, issues=pd.DataFrame(issues))


# ── Part B: feature selection ────────────────────────────────────────────────

def run_partB(ds, cfg, base, splits=None):
    splits = splits or make_splits(ds.y, cfg)
    X, y, genes = ds.X, ds.y, ds.gene_names
    classes = np.arange(len(ds.class_names))
    b = cfg['partB']
    ks = [k for k in b['k_values'] if k <= X.shape[1]]
    rows, scoring, issues = [], [], []

    for method in b['methods']:
        log.info('Part B: %s, k = %s', method, ks)
        t_start = time.perf_counter()
        try:
            for rep, fold, tr, te in splits:
                seed = cfg['seed'] + rep
                # Score genes on the TRAINING fold only (cached, so every k reuses it).
                _, scpu, swall = gene_scores(method, X[tr], y[tr], seed, b)
                scoring.append(dict(method=method, repeat=rep, fold=fold, select_cpu=scpu, select_wall=swall))
                for k in ks:
                    for cname, clf in make_classifiers(cfg, seed).items():
                        pipe = Pipeline([
                            ('select', TopKSelector(method, k, seed, b)),   # fit on training fold
                            ('scale', make_scaler(cfg)),                     # fit on training fold
                            ('clf', clf),
                        ])
                        pred, proba, t = fit_predict(pipe, X[tr], y[tr], X[te])
                        rows.append(dict(method=method, k=k, classifier=cname, repeat=rep, fold=fold,
                                         select_cpu=scpu, select_wall=swall,
                                         **classification_metrics(y[te], pred, proba, classes), **t))
            log.info('  done in %.1fs', time.perf_counter() - t_start)
        except Exception as e:
            log.warning('  FAILED: %s: %s', type(e).__name__, e)
            issues.append(dict(method=method, status='failed', reason=f'{type(e).__name__}: {e}'))

    df = _time_cols(pd.DataFrame(rows))
    scoring = pd.DataFrame(scoring)
    bsum = base['folds'].groupby('classifier')[['accuracy', 'macro_f1', 'total_cpu']].mean()
    summ = df.groupby(['method', 'k', 'classifier'], sort=False)[
        METRICS + ['select_cpu', 'train_cpu', 'pred_cpu', 'total_cpu', 'total_wall']].mean().reset_index()
    summ['delta_accuracy'] = summ.accuracy - summ.classifier.map(bsum.accuracy)
    summ['delta_macro_f1'] = summ.macro_f1 - summ.classifier.map(bsum.macro_f1)
    summ['speedup_cpu'] = summ.classifier.map(bsum.total_cpu) / summ.total_cpu
    stats_df = pd.concat([compare_to_baseline(df, base['folds'], m, ('method', 'k')) for m in ['accuracy', 'macro_f1']])
    anova = anova_across_methods(df, base['folds'], 'accuracy', ('classifier', 'k'))

    # Interpretation (not evaluation): rank genes on ALL cells to see what each method picks.
    methods = [m for m in b['methods'] if m in set(df.method)]
    full_scores, top_rows = {}, []
    markers_set = immune_markers(cfg)
    for m in methods:
        s, _, _ = gene_scores(m, X, y, cfg['seed'], b)
        full_scores[m] = s
        for rank, i in enumerate(top_k(s, b['top_n_genes']), 1):
            top_rows.append(dict(method=m, rank=rank, gene=genes[i], score=s[i], known_marker=genes[i] in markers_set))
    top = pd.DataFrame(top_rows)
    markers = top.groupby('method').agg(n_markers=('known_marker', 'sum'),
                                        markers=('gene', lambda g: ', '.join(x for x in g if x in markers_set))).reset_index()
    sets = {m: top_k(full_scores[m], b['overlap_k']) for m in methods}
    J = np.array([[jaccard(sets[p], sets[q]) for q in methods] for p in methods])
    stab = selection_stability(X, y, methods, b, cfg['seed'])

    _table(df, cfg, 'partB_results_folds.csv')
    _table(summ, cfg, 'partB_summary.csv')
    _table(scoring, cfg, 'partB_selector_cpu.csv')
    _table(stats_df, cfg, 'partB_stats_vs_baseline.csv')
    _table(anova, cfg, 'partB_anova.csv')
    _table(top, cfg, 'partB_top20_genes.csv')
    _table(markers, cfg, 'partB_marker_hits.csv')
    _table(pd.DataFrame(J, index=methods, columns=methods).reset_index(names='method'), cfg, 'partB_method_jaccard.csv')
    _table(stab, cfg, 'partB_stability.csv')
    if issues:
        _table(pd.DataFrame(issues), cfg, 'partB_failed.csv')

    F, dpi = cfg['figures_dir'], cfg['dpi']
    _try_plot(plots.metric_lines, df, base['folds'], 'k', 'accuracy', methods, F, 'partB_accuracy_vs_k.png', dpi,
              'Number of selected genes k', 'Part B — accuracy vs number of selected genes', logx=True)
    _try_plot(plots.metric_lines, df, base['folds'], 'k', 'macro_f1', methods, F, 'partB_f1_vs_k.png', dpi,
              'Number of selected genes k', 'Part B — macro F1 vs number of selected genes', logx=True)
    _try_plot(plots.fs_heatmaps, summ, methods, F, dpi)
    _try_plot(plots.fs_cpu, summ, scoring, methods, F, dpi)
    _try_plot(plots.top_genes, full_scores, genes, markers_set, methods, b['top_n_genes'], F, dpi)
    _try_plot(plots.square_heatmap, J, methods, f'Agreement between methods (Jaccard of top-{b["overlap_k"]} genes)',
              F, 'partB_method_jaccard.png', dpi, 'Jaccard similarity')
    _try_plot(plots.stability_bars, stab, methods, F, dpi)
    return dict(folds=df, summary=summ, scoring=scoring, stats=stats_df, anova=anova, top=top,
                markers=markers, jaccard=pd.DataFrame(J, index=methods, columns=methods), stability=stab,
                scores=full_scores)


def selection_stability(X, y, methods, b, seed):
    """Bootstrap the cells B times; how similar are the top-k gene sets across resamples?"""
    rng = np.random.default_rng(seed)
    boots = [rng.choice(len(y), len(y), replace=True) for _ in range(b['bootstrap'])]
    rows = []
    for m in methods:
        try:
            sets = [top_k(gene_scores(m, X[i], y[i], seed, b)[0], b['stability_k']) for i in boots]
            js = [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
            rows.append(dict(method=m, k=b['stability_k'], n_bootstrap=len(sets),
                             mean_jaccard=np.mean(js), sd_jaccard=np.std(js)))
            log.info('  stability %-20s Jaccard = %.2f', m, np.mean(js))
        except Exception as e:
            log.warning('  stability %s failed: %s', m, e)
    return pd.DataFrame(rows)


# ── Part C: 2D visualisation ─────────────────────────────────────────────────

def embed_2d(method, ds, cfg, seed=None):
    """Fit `method` to 2 dims on ALL cells. Returns (Z, cpu_seconds, wall_seconds)."""
    seed = cfg['seed'] if seed is None else seed
    red = make_reducer(method, 2, cfg, seed, len(ds.X), ds.X.shape[1])
    return timed(red.fit_transform, ds.X)


def _panel_metrics(Xref, Z, y, cfg):
    a = cfg['partA']
    q = embedding_quality(Xref, Z, y, a['quality_k'], cfg['seed'], a['quality_max_cells'])
    knn = KNeighborsClassifier(**cfg['classifiers']['kNN'])
    cvs = StratifiedKFold(cfg['cv']['folds'], shuffle=True, random_state=cfg['seed'])
    q['knn_accuracy_2d'] = cross_val_score(knn, Z, y, cv=cvs).mean()
    return q


def run_partC(ds, cfg, precomputed=None):
    """2D maps from every reducer, plus 2-gene maps from feature selection."""
    precomputed = precomputed or {}
    X, y = ds.X, ds.y
    Xref = make_scaler(cfg).fit_transform(X)
    rows, panels = [], []
    for method in cfg['partC']['methods']:
        log.info('Part C: %s → 2D', method)
        try:
            Z, cpu, wall = precomputed.get(method) or embed_2d(method, ds, cfg)
            q = _panel_metrics(Xref, Z, y, cfg)
            rows.append(dict(method=method, kind='reduction', transductive=not is_inductive(method),
                             cpu_seconds=cpu, wall_seconds=wall, **q))
            panels.append(dict(title=plots.label(method), Z=Z,
                               caption=f"CPU {cpu:.1f}s · trust {q['trustworthiness']:.2f} · kNN acc {q['knn_accuracy_2d']:.2f}"))
        except Exception as e:
            log.warning('  FAILED: %s: %s', type(e).__name__, e)

    fs_panels = []
    rng = np.random.default_rng(cfg['seed'])
    for method in cfg['partC']['fs_methods']:
        try:
            s, cpu, wall = gene_scores(method, X, y, cfg['seed'], cfg['partB'])
            idx = top_k(s, 2)
            Z = X[:, idx]
            q = _panel_metrics(Xref, Z, y, cfg)
            g1, g2 = ds.gene_names[idx]
            rows.append(dict(method=f'{method}: {g1} vs {g2}', kind='selection', transductive=False,
                             cpu_seconds=cpu, wall_seconds=wall, **q))
            # Tiny jitter only for display: many cells share expression = 0.
            Zj = Z + rng.normal(0, 0.04, Z.shape)
            fs_panels.append(dict(title=f'{method}\n{g1} (x) vs {g2} (y)', Z=Zj,
                                  caption=f"CPU {cpu:.1f}s · trust {q['trustworthiness']:.2f} · kNN acc {q['knn_accuracy_2d']:.2f}"))
        except Exception as e:
            log.warning('  FS panel %s failed: %s', method, e)

    m2d = pd.DataFrame(rows)
    _table(m2d, cfg, 'partC_2d_metrics.csv')
    F, dpi = cfg['figures_dir'], cfg['dpi']
    _try_plot(plots.scatter_grid, panels, y, ds.class_names, F, 'partC_2d_grid_dr.png', dpi,
              'Part C — 2D maps of cell types from each reduction method')
    _try_plot(plots.scatter_grid, fs_panels, y, ds.class_names, F, 'partC_2d_grid_fs.png', dpi,
              'Part C — 2D maps from the top-2 genes of each selection method')
    red = m2d[m2d.kind == 'reduction']
    if not red.empty:
        _try_plot(plots.summary_2d, red, list(red.method), F, dpi)
    return dict(metrics=m2d, panels=panels, fs_panels=fs_panels)
