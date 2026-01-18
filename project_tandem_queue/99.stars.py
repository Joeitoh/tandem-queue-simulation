# analysis/stats.py
from math import sqrt
from scipy import stats
import numpy as np
import pandas as pd

def summarize(means):
    m = means.mean()
    se = means.std(ddof=1) / sqrt(len(means))
    ci = (m - 1.96 * se, m + 1.96 * se)
    return {
        'mean': m,
        'se': se,
        'ci': ci,
        'max': means.max(),
        'min': means.min()
    }

def paired_ttest(x, y):
    t, p = stats.ttest_rel(x, y)
    return t, p

def sig_star_p(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


#T検定の話
def paired_diff_stats(m1, m2):
    """
    対応のある2条件の差の統計量をまとめて返す
    """
    diff = m2 - m1
    N = len(diff)

    mean_diff = diff.mean()
    sd_diff   = diff.std(ddof=1)
    se_diff   = sd_diff / sqrt(N)

    t_crit = stats.t.ppf(0.975, df=N-1)
    ci_low = mean_diff - t_crit * se_diff
    ci_up  = mean_diff + t_crit * se_diff

    t_stat = mean_diff / se_diff
    pval = 2 * (1 - stats.t.cdf(abs(t_stat), df=N-1))

    return {
        'mean_diff': mean_diff,
        'se_diff': se_diff,
        'ci': (ci_low, ci_up),
        't': t_stat,
        'p': pval,
        'max': diff.max(),
        'min': diff.min(),
        'diff': diff
    }


def summarize_results(results):
    rows = []
    for config, v in results.items():
        rows.append({
            'Config': f"{config[0]} then {config[1]}",
            'Mean [s]': v['mean'],
            'SE': v['se'],
            '95%CI_Low': v['ci'][0],
            '95%CI_Up': v['ci'][1],
            'Max': v['max'],
            'Min': v['min']
        })
    return pd.DataFrame(rows)

