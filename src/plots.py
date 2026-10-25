"""All figures. Every function takes tidy DataFrames and writes a PNG at cfg dpi.

Colour rules: one fixed categorical order (identity follows the entity, never its
rank), sequential blue for magnitudes, blue↔red around grey for signed
differences. Lines also differ by marker, and transductive methods are dashed
and tagged "(T)", so nothing relies on colour alone.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#86b6ef', '#2a78d6', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6'])
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
BASE_COLOR = '#52514e'
MARKERS = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']
TRANSDUCTIVE = {'SNE', 't-SNE', 'MDS'}

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold', 'legend.frameon': False,
    'lines.linewidth': 2, 'lines.markersize': 7,
})


def palette(names):
    """Fixed colour per name in the order given (falls back to tab20 past 8)."""
    names = list(names)
    cols = CAT if len(names) <= len(CAT) else [plt.cm.tab20(i % 20) for i in range(len(names))]
    return {n: cols[i] for i, n in enumerate(names)}


def label(method):
    return f'{method} (T)' if method in TRANSDUCTIVE else method


def _save(fig, out, name, dpi):
    path = Path(out) / name
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Data summary ─────────────────────────────────────────────────────────────

def class_counts(summary, out, dpi):
    counts = summary['counts']
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(counts.index[::-1], counts.values[::-1], color=CAT[0], height=0.7)
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v + counts.max() * 0.01, i, str(v), va='center', color=INK2, fontsize=9)
    ax.set_xlabel('Number of cells')
    ax.set_title(f"Cells per type — {summary['cells']} cells, {summary['genes']} genes, "
                 f"{summary['classes']} types (imbalance {summary['imbalance_ratio']:.0f}×)")
    ax.grid(axis='y', visible=False)
    return _save(fig, out, 'data_class_counts.png', dpi)


# ── Baseline ─────────────────────────────────────────────────────────────────

def baseline_bars(df, out, dpi):
    """Accuracy and macro-F1 per classifier (mean ± sd over folds), and CPU time."""
    clfs = list(df.classifier.unique())
    g = df.groupby('classifier')[['accuracy', 'macro_f1', 'train_cpu', 'pred_cpu']].agg(['mean', 'std']).reindex(clfs)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(clfs))
    w = 0.38
    for i, (m, title) in enumerate([('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1')]):
        ax.bar(x + (i - 0.5) * w, g[(m, 'mean')], w - 0.02, yerr=g[(m, 'std')], capsize=4,
               color=CAT[i], label=title, error_kw=dict(ecolor=INK2, lw=1))
    ax.set_xticks(x, clfs)
    lo = max(0, (g[('macro_f1', 'mean')] - 3 * g[('macro_f1', 'std')]).min() - 0.05)
    ax.set_ylim(lo, 1)
    ax.set_ylabel('Score (mean ± sd over CV folds)')
    ax.set_title('Baseline: all genes, no reduction')
    ax.legend(loc='lower right')
    p1 = _save(fig, out, 'baseline_accuracy_f1.png', dpi)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (m, title) in enumerate([('train_cpu', 'Training'), ('pred_cpu', 'Prediction')]):
        ax.bar(x + (i - 0.5) * w, g[(m, 'mean')], w - 0.02, yerr=g[(m, 'std')], capsize=4,
               color=CAT[i], label=title, error_kw=dict(ecolor=INK2, lw=1))
    ax.set_xticks(x, clfs)
    ax.set_ylabel('CPU seconds per fold')
    ax.set_title('Baseline: CPU time per fold (all genes)')
    ax.legend()
    p2 = _save(fig, out, 'baseline_cpu_time.png', dpi)
    return p1, p2


# ── Line plots shared by Part A (x = dims) and Part B (x = k genes) ──────────

def metric_lines(df, base, x, metric, methods, out, name, dpi, xlabel, title, logx=False):
    """One panel per classifier, one line per method, baseline as a dashed grey line."""
    clfs = list(base.classifier.unique())
    pal = palette(methods)
    fig, axes = plt.subplots(1, len(clfs), figsize=(5 * len(clfs), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, clf in zip(axes, clfs):
        sub = df[df.classifier == clf]
        for i, m in enumerate(methods):
            s = sub[sub.method == m].groupby(x)[metric].agg(['mean', 'std']).reset_index()
            if s.empty:
                continue
            ls = '--' if m in TRANSDUCTIVE else '-'
            ax.errorbar(s[x], s['mean'], yerr=s['std'], color=pal[m], marker=MARKERS[i % 8],
                        ls=ls, capsize=3, label=label(m), elinewidth=1)
        b = base.loc[base.classifier == clf, metric].mean()
        ax.axhline(b, color=BASE_COLOR, ls=':', lw=1.5, label='Baseline (all genes)')
        ax.set_title(clf)
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale('log')
            ticks = sorted(df[x].unique())
            ax.set_xticks(ticks, [str(t) for t in ticks])
            ax.minorticks_off()
    axes[0].set_ylabel(metric.replace('_', ' '))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 5), bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    return _save(fig, out, name, dpi)


def quality_lines(q, methods, out, dpi):
    cols = [('trustworthiness', 'Trustworthiness'), ('continuity', 'Continuity'),
            ('knn_preservation', 'k-NN preservation'), ('ari', 'k-means ARI'), ('silhouette', 'Silhouette')]
    pal = palette(methods)
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.8))
    for ax, (c, t) in zip(axes, cols):
        for i, m in enumerate(methods):
            s = q[q.method == m].groupby('dim')[c].mean()
            if s.empty:
                continue
            ax.plot(s.index, s.values, color=pal[m], marker=MARKERS[i % 8],
                    ls='--' if m in TRANSDUCTIVE else '-', label=label(m))
        ax.set_title(t)
        ax.set_xlabel('Target dimensions')
        ax.set_xscale('log')
        ticks = sorted(q.dim.unique())
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(l), bbox_to_anchor=(0.5, -0.1))
    fig.suptitle('Embedding quality vs dimensions (higher is better)', fontweight='bold')
    fig.tight_layout()
    return _save(fig, out, 'partA_quality_vs_dim.png', dpi)


def cpu_and_speedup(summary, methods, out, dpi):
    """Left: reducer CPU (fit + transform) vs dims. Right: total-CPU speedup vs baseline."""
    pal = palette(methods)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, m in enumerate(methods):
        s = summary[summary.method == m].groupby('dim').agg(
            reducer=('reducer_cpu', 'mean'), speedup=('speedup_cpu', 'mean'))
        if s.empty:
            continue
        kw = dict(color=pal[m], marker=MARKERS[i % 8], ls='--' if m in TRANSDUCTIVE else '-', label=label(m))
        axes[0].plot(s.index, s.reducer, **kw)
        axes[1].plot(s.index, s.speedup, **kw)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Reducer CPU seconds (fit + transform), log scale')
    axes[0].set_title('How expensive is each reducer?')
    axes[1].axhline(1, color=BASE_COLOR, ls=':', lw=1.5, label='Same cost as baseline')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Speed-up = baseline CPU / method CPU (log)')
    axes[1].set_title('Total pipeline CPU vs baseline (>1 = faster)')
    for ax in axes:
        ax.set_xlabel('Target dimensions')
        ax.set_xscale('log')
        ticks = sorted(summary.dim.unique())
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=min(len(l), 5), bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return _save(fig, out, 'partA_cpu_time.png', dpi)


def delta_heatmap(summary, methods, out, dpi, metric='delta_accuracy'):
    """Method × dims grid of (method − baseline), one panel per classifier."""
    clfs = list(summary.classifier.unique())
    vmax = max(np.nanmax(np.abs(summary[metric].values)), 1e-3)
    dims = sorted(summary.dim.unique())
    fig, axes = plt.subplots(1, len(clfs), figsize=(1.1 * len(dims) * len(clfs) + 3.5, 0.45 * len(methods) + 1.8),
                             sharey=True)
    for i, (ax, clf) in enumerate(zip(np.atleast_1d(axes), clfs)):
        p = summary[summary.classifier == clf].pivot_table(index='method', columns='dim', values=metric)
        p = p.reindex(index=methods, columns=dims)
        im = ax.imshow(p.values, cmap=DIV, vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(p.shape[1]), p.columns)
        ax.set_yticks(range(p.shape[0]), [label(m) for m in p.index])
        ax.tick_params(axis='y', labelleft=(i == 0))
        ax.grid(False)
        for (r, c), v in np.ndenumerate(p.values):
            if np.isnan(v):
                ax.text(c, r, 'skipped', ha='center', va='center', fontsize=7, color=MUTED)
            else:
                ax.text(c, r, f'{v:+.3f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(v) > 0.6 * vmax else INK)
        ax.set_title(clf)
        ax.set_xlabel('Target dimensions')
    fig.colorbar(im, ax=axes, shrink=0.8, label=f'{metric.replace("_", " ")} (method − baseline)')
    fig.suptitle('Change in accuracy vs the all-gene baseline (blue = better, red = worse)', fontweight='bold')
    return _save(fig, out, 'partA_delta_accuracy_heatmap.png', dpi)


# ── Part B extras ────────────────────────────────────────────────────────────

def fs_heatmaps(summary, methods, out, dpi, metric='accuracy'):
    """Method × classifier heatmap of mean accuracy, one panel per k."""
    ks = sorted(summary.k.unique())
    vmin, vmax = summary[metric].min(), summary[metric].max()
    fig, axes = plt.subplots(1, len(ks), figsize=(3.4 * len(ks) + 1, 0.5 * len(methods) + 1.8), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), ks):
        p = summary[summary.k == k].pivot_table(index='method', columns='classifier', values=metric)
        p = p.reindex(index=[m for m in methods if m in p.index],
                      columns=[c for c in ['kNN', 'SVM', 'RF'] if c in p.columns])
        im = ax.imshow(p.values, cmap=SEQ, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(p.shape[1]), p.columns)
        ax.set_yticks(range(p.shape[0]), p.index)
        ax.grid(False)
        for (r, c), v in np.ndenumerate(p.values):
            ax.text(c, r, f'{v:.3f}', ha='center', va='center', fontsize=8,
                    color='white' if v > vmin + 0.6 * (vmax - vmin) else INK)
        ax.set_title(f'k = {k} genes')
    fig.colorbar(im, ax=axes, shrink=0.8, label=f'Mean {metric}')
    fig.suptitle(f'{metric.capitalize()}: selection method × classifier', fontweight='bold')
    return _save(fig, out, 'partB_heatmap_accuracy.png', dpi)


def fs_cpu(summary, scoring, methods, out, dpi):
    pal = palette(methods)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    s = scoring.groupby('method')['select_cpu'].agg(['mean', 'std']).reindex(methods).dropna(how='all')
    axes[0].barh([m for m in s.index][::-1], s['mean'][::-1], xerr=s['std'][::-1],
                 color=[pal[m] for m in s.index][::-1], capsize=3, error_kw=dict(ecolor=INK2, lw=1))
    axes[0].set_xscale('log')
    axes[0].set_xlabel('CPU seconds to score all genes (one fold, log)')
    axes[0].set_title('Cost of the selector itself')
    axes[0].grid(axis='y', visible=False)
    for i, m in enumerate(methods):
        t = summary[summary.method == m].groupby('k')['total_cpu'].mean()
        if not t.empty:
            axes[1].plot(t.index, t.values, color=pal[m], marker=MARKERS[i % 8], label=m)
    axes[1].set_xscale('log')
    ks = sorted(summary.k.unique())
    axes[1].set_xticks(ks, [str(k) for k in ks])
    axes[1].minorticks_off()
    axes[1].set_xlabel('Number of selected genes k')
    axes[1].set_ylabel('CPU s (select + train + predict), mean of classifiers')
    axes[1].set_title('Total pipeline cost')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out, 'partB_cpu_time.png', dpi)


def top_genes(scores, genes, markers, methods, n, out, dpi):
    """Top-n genes per method; known immune markers are starred and coloured."""
    methods = [m for m in methods if m in scores]
    cols = min(len(methods), 3)
    rows = int(np.ceil(len(methods) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 0.24 * n * rows + 1.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, methods):
        s = scores[m]
        idx = np.argsort(-s, kind='stable')[:n][::-1]
        vals = s[idx] / (s.max() + 1e-12)
        names = [f'★ {genes[i]}' if genes[i] in markers else genes[i] for i in idx]
        colors = [CAT[1] if genes[i] in markers else CAT[0] for i in idx]
        ax.barh(names, vals, color=colors, height=0.75)
        ax.set_title(m)
        ax.set_xlabel('Score / max score')
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', visible=False)
    for ax in axes[len(methods):]:
        ax.axis('off')
    fig.suptitle(f'Top {n} genes per method — ★ orange = known immune marker', fontweight='bold')
    fig.tight_layout()
    return _save(fig, out, 'partB_top20_genes.png', dpi)


def square_heatmap(M, names, title, out, name, dpi, cbar_label):
    fig, ax = plt.subplots(figsize=(0.9 * len(names) + 2.5, 0.75 * len(names) + 1.5))
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30, ha='right')
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for (r, c), v in np.ndenumerate(M):
        ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=9, color='white' if v > 0.6 else INK)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    return _save(fig, out, name, dpi)


def stability_bars(stab, methods, out, dpi):
    pal = palette(methods)
    s = stab.set_index('method').reindex([m for m in methods if m in set(stab.method)])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(s.index[::-1], s.mean_jaccard[::-1], xerr=s.sd_jaccard[::-1], capsize=3,
            color=[pal[m] for m in s.index][::-1], error_kw=dict(ecolor=INK2, lw=1))
    for i, v in enumerate(s.mean_jaccard[::-1]):
        ax.text(min(v + 0.03, 0.95), i, f'{v:.2f}', va='center', color=INK2, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Mean pairwise Jaccard of selected gene sets across bootstraps')
    ax.set_title('Selection stability (1 = the same genes every time)')
    ax.grid(axis='y', visible=False)
    return _save(fig, out, 'partB_stability.png', dpi)


# ── Part C: 2D scatter grids ─────────────────────────────────────────────────

def scatter_ax(ax, Z, y, class_names, title='', caption=''):
    """One 2D map coloured by cell type, with numbered labels at each type's median."""
    pal = palette(class_names)
    for c, cname in enumerate(class_names):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], s=5, color=pal[cname], alpha=0.7, linewidths=0, label=cname)
    # Direct labels so identity never depends on colour alone.
    for c, cname in enumerate(class_names):
        m = y == c
        if m.sum() >= 3:
            cx, cy = np.median(Z[m, 0]), np.median(Z[m, 1])
            ax.text(cx, cy, str(c + 1), fontsize=8, fontweight='bold', ha='center', va='center',
                    color=INK, bbox=dict(boxstyle='circle,pad=0.15', fc='white', ec=pal[cname], lw=1.2, alpha=0.85))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(GRID)
    ax.set_xlabel(caption, fontsize=8.5, color=INK2)


def legend_handles(class_names):
    pal = palette(class_names)
    return [plt.Line2D([], [], marker='o', ls='', color=pal[c], markersize=7, label=f'{i + 1}  {c}')
            for i, c in enumerate(class_names)]


def scatter_grid(panels, y, class_names, out, name, dpi, title):
    """panels: list of dict(title, Z, caption). Same colours in every panel, shared legend."""
    n = len(panels)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    legend_in = 1.2
    height = 4.6 * rows + legend_in
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, height))
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, panels):
        scatter_ax(ax, p['Z'], y, class_names, p['title'], p.get('caption', ''))
    for ax in axes[n:]:
        ax.axis('off')
    handles = legend_handles(class_names)
    fig.suptitle(title, fontweight='bold', fontsize=13)
    fig.tight_layout(rect=(0, legend_in / height, 1, 1))
    fig.legend(handles=handles, loc='lower center', ncol=min(len(class_names), 4),
               bbox_to_anchor=(0.5, 0), title='Cell type (number = label at the type median)')
    return _save(fig, out, name, dpi)


def summary_2d(m2d, methods, out, dpi):
    pal = palette(methods)
    cols = [('trustworthiness', 'Trustworthiness ↑', False), ('continuity', 'Continuity ↑', False),
            ('ari', 'k-means ARI ↑', False), ('cpu_seconds', 'CPU seconds ↓ (log)', True)]
    s = m2d.set_index('method').reindex([m for m in methods if m in set(m2d.method)])
    fig, axes = plt.subplots(1, 4, figsize=(17, 0.42 * len(s) + 1.8), sharey=True)
    for ax, (c, t, logx) in zip(axes, cols):
        ax.barh([label(m) for m in s.index][::-1], s[c][::-1], color=[pal[m] for m in s.index][::-1], height=0.7)
        for i, v in enumerate(s[c][::-1]):
            ax.text(v, i, f' {v:.2f}' if not logx else f' {v:.1f}s', va='center', fontsize=8, color=INK2)
        if logx:
            ax.set_xscale('log')
        else:
            ax.set_xlim(min(0, s[c].min()), 1.08)
        ax.set_title(t)
        ax.grid(axis='y', visible=False)
    fig.suptitle('2D methods compared', fontweight='bold')
    fig.tight_layout()
    return _save(fig, out, 'partC_2d_summary.png', dpi)
