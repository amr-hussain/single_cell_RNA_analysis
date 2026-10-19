import sys
from pathlib import Path
import numpy as np, pandas as pd
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

def main():
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / 'config.yaml')
    
    print("Loading data...")
    # Load dataset using existing benchmark_sample function
    X, y, train_mask = benchmark_sample(root / 'data', cfg['train_per_class'], cfg['test_per_class'], cfg['random_seed'])
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    
    # Baseline (All Features)
    evaluate_selection(X_train, X_test, y_train, y_test, "Baseline (784 Features)", cfg['classifier_neighbors'])

    # 1. Univariate Feature Selection (ANOVA F-value)
    k_features = 100
    print(f"\nSelecting top {k_features} features using ANOVA...")
    selector_anova = SelectKBest(f_classif, k=k_features)
    X_train_anova = selector_anova.fit_transform(X_train, y_train)
    X_test_anova = selector_anova.transform(X_test)
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