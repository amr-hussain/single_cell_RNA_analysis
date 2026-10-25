import itertools
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.multivariate.manova import MANOVA


def compare_methods(repeat_df):
    methods = repeat_df.method.unique()
    rows = []
    pivot = repeat_df.pivot(index='repeat', columns='method', values='accuracy')

    for a, b in itertools.combinations(methods, 2):
        x = pivot[a].dropna()
        y = pivot[b].dropna()
        common = x.index.intersection(y.index)
        d = x.loc[common] - y.loc[common]

        tt = stats.ttest_rel(x.loc[common], y.loc[common])
        wz = stats.wilcoxon(d) if np.any(d) else SimpleNamespace(statistic=0.0, pvalue=1.0)
        z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)

        rows += [
            dict(test='paired t-test', comparison=f'{a} vs {b}', statistic=tt.statistic, p_value=tt.pvalue),
            dict(test='paired z-test (normal approximation)', comparison=f'{a} vs {b}', statistic=z, p_value=2 * stats.norm.sf(abs(z))),
            dict(test='Wilcoxon signed-rank', comparison=f'{a} vs {b}', statistic=wz.statistic, p_value=wz.pvalue),
        ]

    groups = [repeat_df.loc[repeat_df.method == m, 'accuracy'] for m in methods]
    f, p = stats.f_oneway(*groups)
    rows.append(dict(test='one-way ANOVA', comparison='all methods', statistic=f, p_value=p))
    return pd.DataFrame(rows)


def manova_text(repeat_df):
    safe = repeat_df.copy()
    safe['method_code'] = safe.method.astype('category').cat.codes
    return str(MANOVA.from_formula('accuracy + macro_f1 + auc_roc ~ C(method)', data=safe).mv_test())
