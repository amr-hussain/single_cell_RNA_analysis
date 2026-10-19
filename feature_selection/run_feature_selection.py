import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Add parent directory to path to use existing src modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data import benchmark_sample
from src.config import load_config

def evaluate_selection(X_train, X_test, y_train, y_test, method_name, k=7):
    clf = KNeighborsClassifier(k, weights='distance').fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"[{method_name}] k-NN Accuracy: {acc:.4f}")

def plot_anova_scores(scores, feature_names, top_n, output_dir):
    """Generates a horizontal bar chart of ANOVA scores as shown in the lecture."""
    # Handle NaN scores which can happen with zero-variance pixels
    scores = np.nan_to_num(scores, nan=0.0)
    
    # Get indices of the top N features
    top_indices = np.argsort(scores)[-top_n:]
    top_scores = scores[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(top_names, top_scores, color='#1f77b4')
    plt.xlabel('ANOVA F-Value Score')
    plt.ylabel('Pixel Index')
    plt.title(f'Top {top_n} Features by ANOVA Score')
    plt.tight_layout()
    
    output_path = output_dir / 'anova_feature_scores.png'
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved ANOVA scores visualization to {output_path}")

def main():
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / 'config.yaml')
    
    # Ensure output directory exists
    fig_dir = root / 'outputs' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    X, y, train_mask = benchmark_sample(root / 'data', cfg['train_per_class'], cfg['test_per_class'], cfg['random_seed'])
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    
    # Create feature names (Pixel_0 to Pixel_783 for Fashion-MNIST)
    feature_names = [f"Pixel_{i}" for i in range(X_train.shape[1])]
    
    # Baseline (All Features)
    evaluate_selection(X_train, X_test, y_train, y_test, "Baseline (784 Features)", cfg['classifier_neighbors'])

    # 1. Univariate Feature Selection (ANOVA F-value)
    k_features = 100
    print(f"\nSelecting top {k_features} features using ANOVA...")
    
    selector_anova = SelectKBest(score_func=f_classif, k=k_features)
    X_train_anova = selector_anova.fit_transform(X_train, y_train)
    X_test_anova = selector_anova.transform(X_test)
    
    # Extract scores and plot the top 20 as a horizontal bar chart
    plot_anova_scores(selector_anova.scores_, feature_names, top_n=20, output_dir=fig_dir)
    
    evaluate_selection(X_train_anova, X_test_anova, y_train, y_test, f"ANOVA SelectKBest (k={k_features})", cfg['classifier_neighbors'])

    # 2. Model-Based Feature Selection (Random Forest)
    print("\nSelecting features based on Random Forest importance...")
    rf = RandomForestClassifier(n_estimators=50, random_state=cfg['random_seed'])
    rf.fit(X_train, y_train)
    
    selector_rf = SelectFromModel(rf, prefit=True, max_features=k_features)
    X_train_rf = selector_rf.transform(X_train)
    X_test_rf = selector_rf.transform(X_test)
    evaluate_selection(X_train_rf, X_test_rf, y_train, y_test, f"Random Forest Selection ({X_train_rf.shape[1]} features)", cfg['classifier_neighbors'])

if __name__ == '__main__':
    main()