import numpy as np
import pandas as pd
from math import sqrt
from scipy import stats

# *******************************************************************************
# 各場所での待ち行列長 Lq を計測
# *******************************************************************************
# ********** 関数定義 *****************
def run_Lq_simulation(staff_stage1, staff_stage2, n_reps=500):
    Lq1 = np.empty(n_reps)
    Lq2 = np.empty(n_reps)

    for r in range(n_reps):
        rs = np.random.RandomState(SEED + r)

        interarr = rs.exponential(MEAN_INTERARRIVAL, size=N_CUSTOMERS)
        arrivals = np.cumsum(interarr)
        service_B = rs.exponential(MEAN_SERVICE_B, size=N_CUSTOMERS)

        s1 = np.full(N_CUSTOMERS, SERVICE_A) if staff_stage1 == 'A' else service_B
        s2 = np.full(N_CUSTOMERS, SERVICE_A) if staff_stage2 == 'A' else service_B

        start1 = np.empty(N_CUSTOMERS)
        finish1 = np.empty(N_CUSTOMERS)
        start2 = np.empty(N_CUSTOMERS)
        finish2 = np.empty(N_CUSTOMERS)

        t1 = t2 = 0.0
        for i in range(N_CUSTOMERS):
            start1[i] = max(arrivals[i], t1)
            finish1[i] = start1[i] + s1[i]
            t1 = finish1[i]

            start2[i] = max(finish1[i], t2)
            finish2[i] = start2[i] + s2[i]
            t2 = finish2[i]

        Wq1 = np.maximum(0, start1 - arrivals)
        Wq2 = np.maximum(0, start2 - finish1)

        T = finish2[-1] - arrivals[0]

        Lq1[r] = Wq1.sum() / T
        Lq2[r] = Wq2.sum() / T

    return {'Wait1': Lq1, 'Wait2': Lq2}

# シミュレーション----Now Let's go--------------------------------
res_I  = run_Lq_simulation('A', 'B', n_reps=500)
res_II = run_Lq_simulation('B', 'A', n_reps=500)

# 結果表として整形
rows = []
for metric in ['Wait1', 'Wait2']:
    data_I  = res_I[metric]
    data_II = res_II[metric]

    # 対応のある t 検定（CRN 前提）
    t_stat, pval = stats.ttest_rel(data_II, data_I)
    star = sig_star_p(pval)

    for label, data in [('条件Ⅰ (A–B)', data_I), ('条件Ⅱ (B–A)', data_II)]:
        mean = data.mean()
        se = data.std(ddof=1) / np.sqrt(len(data))
        ci_low, ci_up = stats.t.interval(
            0.95, len(data)-1, loc=mean, scale=se
        )

        rows.append({
            '条件': label,
            '項目': metric,
            '平均値': mean,
            '標準誤差': se,
            '95%CI_Low': ci_low,
            '95%CI_Up': ci_up,
            'MAX': data.max(),
            'MIN': data.min(),
            'Sig': star if 'Ⅱ' in label else ""  # ← 条件Ⅱだけスター
        })

df = pd.DataFrame(rows)

print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))