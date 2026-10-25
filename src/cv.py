"""Cross-validation helpers shared by the baseline, Part A and Part B."""
import logging
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from .metrics import classification_metrics

log = logging.getLogger('tutorial')


def make_classifiers(cfg, seed):
    c = cfg['classifiers']
    return {
        'kNN': KNeighborsClassifier(**c['kNN']),
        'SVM': SVC(kernel='rbf', random_state=seed, cache_size=500, **c['SVM']),
        'RF': RandomForestClassifier(random_state=seed, **c['RF']),
    }


def make_splits(y, cfg):
    """The SAME stratified folds for every method → results can be paired fold by fold."""
    splits = []
    for rep in range(cfg['cv']['repeats']):
        skf = StratifiedKFold(cfg['cv']['folds'], shuffle=True, random_state=cfg['seed'] + rep)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            splits.append((rep, fold, tr, te))
    return splits


def fit_predict(model, Xtr, ytr, Xte):
    """Fit + predict with CPU (process_time) and wall (perf_counter) timings."""
    t0, w0 = time.process_time(), time.perf_counter()
    model.fit(Xtr, ytr)
    fit_cpu, fit_wall = time.process_time() - t0, time.perf_counter() - w0
    t0, w0 = time.process_time(), time.perf_counter()
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)
    pred_cpu, pred_wall = time.process_time() - t0, time.perf_counter() - w0
    return pred, proba, dict(train_cpu=fit_cpu, train_wall=fit_wall, pred_cpu=pred_cpu, pred_wall=pred_wall)


def eval_classifiers(Ztr, ytr, Zte, yte, cfg, seed, classes):
    """Train every classifier on the given features; one result row per classifier."""
    rows = []
    for name, clf in make_classifiers(cfg, seed).items():
        try:
            pred, proba, t = fit_predict(clf, Ztr, ytr, Zte)
            row = dict(classifier=name, **classification_metrics(yte, pred, proba, classes), **t)
            rows.append(row)
        except Exception as e:  # keep going if one classifier fails
            log.warning('    %s failed: %s', name, e)
    return rows
