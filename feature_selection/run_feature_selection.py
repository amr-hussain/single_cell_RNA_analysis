"""
Feature Selection + Classification Benchmark on Fashion-MNIST
=============================================================

Feature selection methods:
    1. No Feature Selection (Baseline)
    2. Variance Threshold
    3. ANOVA F-test
    4. Mutual Information
    5. Fisher Score

Classifiers:
    KNN, SVM, Random Forest

Metrics:
    Accuracy, Precision, Recall, F1 Score, Training Time
"""

import sys, time, warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to use existing src modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data import benchmark_sample, load_fashion_mnist
from src.config import load_config

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', context='talk')


# ────────────────────────────── Fisher Score ──────────────────────────────

def fisher_score(X, y):
    """
    Compute Fisher Score for each feature.

    Fisher Score = Σ_c  n_c * (μ_c - μ)^2 / Σ_c  n_c * σ_c^2

    Higher is better (more discriminative).
    """
    classes = np.unique(y)
    n_features = X.shape[1]
    overall_mean = X.mean(axis=0)
    numerator = np.zeros(n_features)
    denominator = np.zeros(n_features)

    for c in classes:
        Xc = X[y == c]
        nc = len(Xc)
        class_mean = Xc.mean(axis=0)
        class_var = Xc.var(axis=0)
        numerator += nc * (class_mean - overall_mean) ** 2
        denominator += nc * class_var

    denominator[denominator == 0] = 1e-12  # avoid division by zero
    return numerator / denominator


# ────────────────────────── Feature Selectors ─────────────────────────────

def select_no_selection(X_train, X_test, y_train, k):
    """Baseline: return all features."""
    return X_train, X_test, "No Selection (Baseline)", X_train.shape[1]


def select_variance_threshold(X_train, X_test, y_train, k):
    """Remove features with variance below a computed threshold to keep ~k."""
    variances = np.var(X_train, axis=0)
    sorted_var = np.sort(variances)[::-1]
    if k < len(sorted_var):
        threshold = sorted_var[k - 1]
    else:
        threshold = 0.0
    # Use a slightly lower threshold to ensure we get at least k features
    sel = VarianceThreshold(threshold=max(threshold * 0.99, 0.0))
    X_tr = sel.fit_transform(X_train)
    X_te = sel.transform(X_test)
    # If we got more features than k, manually pick top-k by variance
    if X_tr.shape[1] > k:
        remaining_var = np.var(X_tr, axis=0)
        top_k_idx = np.argsort(remaining_var)[-k:]
        X_tr = X_tr[:, top_k_idx]
        X_te = X_te[:, top_k_idx]
    n_sel = X_tr.shape[1]
    return X_tr, X_te, f"Variance Threshold", n_sel


def select_anova(X_train, X_test, y_train, k):
    """ANOVA F-test: select top k features."""
    sel = SelectKBest(score_func=f_classif, k=k)
    X_tr = sel.fit_transform(X_train, y_train)
    X_te = sel.transform(X_test)
    return X_tr, X_te, "ANOVA F-test", k


def select_mutual_info(X_train, X_test, y_train, k):
    """Mutual information: select top k features."""
    sel = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=42),
        k=k
    )
    X_tr = sel.fit_transform(X_train, y_train)
    X_te = sel.transform(X_test)
    return X_tr, X_te, "Mutual Information", k


def select_fisher(X_train, X_test, y_train, k):
    """Fisher Score: select top k features."""
    scores = fisher_score(X_train, y_train)
    top_k_idx = np.argsort(scores)[-k:]
    X_tr = X_train[:, top_k_idx]
    X_te = X_test[:, top_k_idx]
    return X_tr, X_te, "Fisher Score", k


# ────────────────────────── Classifiers ───────────────────────────────────

def get_classifiers(seed):
    return {
        'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=seed, cache_size=1000),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1
        ),
    }


# ────────────────────── Evaluation Pipeline ───────────────────────────────

def evaluate_pipeline(X_train, X_test, y_train, y_test, clf_name, clf):
    """Train classifier and return metrics + training & prediction time."""
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = clf.predict(X_test)
    pred_time = time.perf_counter() - t1

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average='macro', zero_division=0)
    rec = recall_score(y_test, pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, pred, average='macro', zero_division=0)

    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'Training Time (s)': train_time,
        'Prediction Time (s)': pred_time,
        'Total Time (s)': train_time + pred_time,
    }


# ────────────────────── Visualization Functions ───────────────────────────

def plot_comparison_heatmap(df, metric, output_dir):
    """Heatmap: feature selection method × classifier for a given metric."""
    pivot = df.pivot_table(
        index='Feature Selection',
        columns='Classifier',
        values=metric,
        aggfunc='mean',
    )
    # Consistent ordering
    fs_order = [
        'No Selection (Baseline)',
        'Variance Threshold',
        'ANOVA F-test',
        'Mutual Information',
        'Fisher Score',
    ]
    clf_order = ['KNN', 'SVM', 'Random Forest']
    pivot = pivot.reindex(
        index=[f for f in fs_order if f in pivot.index],
        columns=[c for c in clf_order if c in pivot.columns],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.4f',
        cmap='YlGnBu',
        ax=ax,
        linewidths=0.5,
        linecolor='white',
    )
    ax.set_title(f'{metric} — Feature Selection × Classifier', fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    slug = metric.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(output_dir / f'fs_heatmap_{slug}.png', dpi=180)
    plt.close()


def plot_grouped_bars(df, metric, output_dir):
    """Grouped bar chart comparing classifiers across feature selection methods."""
    fs_order = [
        'No Selection (Baseline)',
        'Variance Threshold',
        'ANOVA F-test',
        'Mutual Information',
        'Fisher Score',
    ]
    df_plot = df[df['Feature Selection'].isin(fs_order)].copy()
    df_plot['Feature Selection'] = pd.Categorical(
        df_plot['Feature Selection'], categories=fs_order, ordered=True
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=df_plot,
        x='Feature Selection',
        y=metric,
        hue='Classifier',
        ax=ax,
        palette='viridis',
        edgecolor='black',
    )
    ax.set_title(
        f'{metric} by Feature Selection Method & Classifier',
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    ax.legend(title='Classifier', loc='upper right')
    plt.tight_layout()
    slug = metric.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(output_dir / f'fs_bars_{slug}.png', dpi=180)
    plt.close()


def plot_radar(df, output_dir):
    """Radar chart comparing classifiers under each feature selection method."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    fs_methods = df['Feature Selection'].unique()
    clfs = df['Classifier'].unique()
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(clfs)))

    n_methods = len(fs_methods)
    fig, axes = plt.subplots(
        1,
        n_methods,
        figsize=(5 * n_methods, 5),
        subplot_kw=dict(polar=True),
    )
    if n_methods == 1:
        axes = [axes]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for ax, fs in zip(axes, fs_methods):
        for clf, color in zip(clfs, colors):
            row = df[(df['Feature Selection'] == fs) & (df['Classifier'] == clf)]
            if row.empty:
                continue
            values = row[metrics].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=clf, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(fs, fontsize=10, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'fs_radar_chart.png', dpi=180)
    plt.close()


def plot_training_time(df, output_dir):
    """Bar chart of training time."""
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=df,
        x='Feature Selection',
        y='Training Time (s)',
        hue='Classifier',
        ax=ax,
        palette='magma',
        edgecolor='black',
    )
    ax.set_title(
        'Training Time by Feature Selection × Classifier on Core I5 11300h',
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    ax.legend(title='Classifier')
    plt.tight_layout()
    plt.savefig(output_dir / 'fs_training_time.png', dpi=180)
    plt.close()


def plot_prediction_time(df, output_dir):
    """Bar chart of prediction / inference time."""
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=df,
        x='Feature Selection',
        y='Prediction Time (s)',
        hue='Classifier',
        ax=ax,
        palette='viridis',
        edgecolor='black',
    )
    ax.set_title(
        'Prediction Time by Feature Selection × Classifier on Core I5 11300h',
        fontsize=14,
        fontweight='bold',
    )

    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    ax.legend(title='Classifier')
    plt.tight_layout()
    plt.savefig(output_dir / 'fs_prediction_time.png', dpi=180)
    plt.close()


def plot_feature_scores(X_train, y_train, output_dir, top_n=30):
    """Visualize feature scores from ANOVA, MI, and Fisher side by side."""
    n_features = X_train.shape[1]
    feature_names = [f"Px{i}" for i in range(n_features)]

    # Compute all scores
    anova_scores = f_classif(X_train, y_train)[0]
    anova_scores = np.nan_to_num(anova_scores, nan=0.0)
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    f_scores = fisher_score(X_train, y_train)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    for ax, scores, title in zip(
        axes,
        [anova_scores, mi_scores, f_scores],
        ['ANOVA F-value', 'Mutual Information', 'Fisher Score']
    ):
        top_idx = np.argsort(scores)[-top_n:]
        top_vals = scores[top_idx]
        top_names = [feature_names[i] for i in top_idx]
        ax.barh(top_names, top_vals, color='#2196F3', edgecolor='white')
        ax.set_xlabel('Score')
        ax.set_title(f'Top {top_n} — {title}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fs_feature_scores_comparison.png', dpi=180)
    plt.close()


def plot_selected_pixels(X_train, y_train, output_dir, k=100):
    """Show which pixels are selected by each method on a 28×28 grid."""
    n_features = X_train.shape[1]
    img_side = int(np.sqrt(n_features))

    # ANOVA
    anova_scores = f_classif(X_train, y_train)[0]
    anova_scores = np.nan_to_num(anova_scores, nan=0.0)
    anova_mask = np.zeros(n_features)
    anova_mask[np.argsort(anova_scores)[-k:]] = 1

    # Mutual Information
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_mask = np.zeros(n_features)
    mi_mask[np.argsort(mi_scores)[-k:]] = 1

    # Fisher Score
    f_scores = fisher_score(X_train, y_train)
    fisher_mask = np.zeros(n_features)
    fisher_mask[np.argsort(f_scores)[-k:]] = 1

    # Variance
    variances = np.var(X_train, axis=0)
    var_mask = np.zeros(n_features)
    var_mask[np.argsort(variances)[-k:]] = 1

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, mask, title in zip(
        axes,
        [var_mask, anova_mask, mi_mask, fisher_mask],
        ['Variance Threshold', 'ANOVA F-test', 'Mutual Information', 'Fisher Score']
    ):
        ax.imshow(
            mask.reshape(img_side, img_side),
            cmap='hot',
            interpolation='nearest',
        )
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle(
        f'Top {k} Selected Pixels by Each Method',
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / 'fs_selected_pixels.png',
        dpi=180,
        bbox_inches='tight',
    )
    plt.close()


# ────────────────────────── Main ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Feature Selection + Classification Benchmark")
    parser.add_argument('--full', action='store_true', help="Use the FULL Fashion-MNIST dataset (60,000 train, 10,000 test)")
    parser.add_argument('--output-dir', type=str, default=None, help="Custom output root directory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / 'config.yaml')
    seed = cfg['random_seed']

    # Output directories: separate directory for full dataset vs sample benchmark
    if args.output_dir:
        base_out = Path(args.output_dir)
    elif args.full:
        base_out = root / 'outputs' / 'full_dataset'
    else:
        base_out = root / 'outputs'

    fig_dir = base_out / 'figures'
    tab_dir = base_out / 'tables'
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    dataset_title = "FULL Dataset (60k/10k)" if args.full else "Balanced Sample Subset"
    print(f"  Feature Selection + Classification Benchmark — {dataset_title}")
    print(f"  Output directory: {base_out}")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    if args.full:
        Xtr, ytr, Xte, yte = load_fashion_mnist(root / 'data')
        X_train = (Xtr / 255.0).astype(np.float32)
        y_train = ytr
        X_test = (Xte / 255.0).astype(np.float32)
        y_test = yte
    else:
        X, y, train_mask = benchmark_sample(
            root / 'data', cfg['train_per_class'], cfg['test_per_class'], seed
        )
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]
    print(f"      Train: {X_train.shape}, Test: {X_test.shape}")

    # Scale data (important for SVM)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Number of features to select
    k = 100

    # Feature selection methods
    selectors = [
        select_no_selection,
        select_variance_threshold,
        select_anova,
        select_mutual_info,
        select_fisher,
    ]

    # Run all combinations
    print(f"\n[2/5] Running {len(selectors)} selectors × 3 classifiers...")
    results = []
    for sel_fn in selectors:
        X_tr, X_te, fs_name, n_sel = sel_fn(X_train_sc, X_test_sc, y_train, k)
        print(f"\n  ── {fs_name} ({n_sel} features) ──")

        for clf_name, clf in get_classifiers(seed).items():
            metrics = evaluate_pipeline(X_tr, X_te, y_train, y_test,
                                        clf_name, clf)
            metrics['Feature Selection'] = fs_name
            metrics['Classifier'] = clf_name
            metrics['Num Features'] = n_sel
            results.append(metrics)
            log_msg = (
                f"    {clf_name:15s}  Acc={metrics['Accuracy']:.4f}  "
                f"F1={metrics['F1 Score']:.4f}  "
                f"TrainTime={metrics['Training Time (s)']:.3f}s  "
                f"PredTime={metrics['Prediction Time (s)']:.3f}s"
            )
            print(log_msg)

    df = pd.DataFrame(results)

    # Save results table
    print(f"\n[3/5] Saving results...")
    csv_path = tab_dir / 'feature_selection_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"      Saved to {csv_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    display_cols = ['Feature Selection', 'Classifier', 'Num Features',
                    'Accuracy', 'Precision', 'Recall', 'F1 Score',
                    'Training Time (s)', 'Prediction Time (s)']
    print(df[display_cols].to_string(index=False))

    # Best combination
    best_msg = (
        f"\n  ★ Best Accuracy: {best['Accuracy']:.4f} — "
        f"{best['Classifier']} with {best['Feature Selection']}"
    )
    print(best_msg)

    # Generate visualizations
    print(f"\n[4/5] Generating plots...")

    # Feature score visualizations
    # If full dataset, compute scores on representative subset to avoid memory/time bottlenecks
    score_train_sc = X_train_sc[:10000] if args.full else X_train_sc
    score_y = y_train[:10000] if args.full else y_train
    score_train_unscaled = X_train[:10000] if args.full else X_train

    plot_feature_scores(score_train_sc, score_y, fig_dir)
    print("      ✓ Feature scores comparison")

    plot_selected_pixels(score_train_unscaled, score_y, fig_dir, k=k)
    print("      ✓ Selected pixel maps")

    # Heatmaps for each metric
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        plot_comparison_heatmap(df, metric, fig_dir)
    print("      ✓ Metric heatmaps")

    # Grouped bar charts
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        plot_grouped_bars(df, metric, fig_dir)
    print("      ✓ Grouped bar charts")

    # Radar chart
    plot_radar(df, fig_dir)
    print("      ✓ Radar chart")

    # Training and Prediction times
    plot_training_time(df, fig_dir)
    print("      ✓ Training time chart")

    plot_prediction_time(df, fig_dir)
    print("      ✓ Prediction time chart")

    print(f"\n[5/5] Done! All outputs saved in {base_out}")
    print("=" * 70)

    return df


if __name__ == '__main__':
    main()