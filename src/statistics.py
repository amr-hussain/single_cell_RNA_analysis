from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import stats

FOLD_KEYS = ['classifier', 'repeat', 'fold']


def compare_to_baseline(df, base, metric='accuracy', keys=('method', 'dim')):
    """Paired t-test and Wilcoxon signed-rank of each method vs the baseline.

    Pairs are matched on (classifier, repeat, fold), so both sides saw exactly the
    same training/test cells. Caveat: CV folds share training data, so they are not
    fully independent and p-values are somewhat optimistic.
    """
    keys = list(keys)
    b = base.set_index(FOLD_KEYS)[metric]
    rows = []
    for gkey, g in df.groupby(['classifier'] + keys):
        a = g.set_index(FOLD_KEYS)[metric]
        common = a.index.intersection(b.index)
        x, z = a.loc[common], b.loc[common]
        d = x - z
        if len(d) > 1 and d.std(ddof=1) > 0:
            tt = stats.ttest_rel(x, z)
        else:
            tt = SimpleNamespace(statistic=np.nan, pvalue=np.nan)
        wz = stats.wilcoxon(d) if np.any(d) and len(d) > 1 else SimpleNamespace(statistic=0.0, pvalue=1.0)
        rows.append(dict(
            zip(['classifier'] + keys, gkey),
            metric=metric,
            n_pairs=len(d),
            mean_diff_vs_baseline=d.mean(),
            ttest_stat=tt.statistic, ttest_p=tt.pvalue,
            wilcoxon_stat=wz.statistic, wilcoxon_p=wz.pvalue,
        ))
    return pd.DataFrame(rows)


def anova_across_methods(df, base, metric='accuracy', by=('classifier', 'dim')):
    """One-way ANOVA: do the methods (plus the baseline) differ at all, per group?"""
    rows = []
    for gkey, g in df.groupby(list(by)):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        groups = [s[metric].dropna().values for _, s in g.groupby('method')]
        clf = dict(zip(by, gkey))['classifier']
        groups.append(base.loc[base.classifier == clf, metric].values)
        groups = [x for x in groups if len(x) > 1]
        if len(groups) < 2:
            continue
        f, p = stats.f_oneway(*groups)
        rows.append(dict(zip(by, gkey), metric=metric, n_groups=len(groups), F=f, p_value=p))
    return pd.DataFrame(rows)
