#======================================
#  各場所での待ち時間を計測
#======================================

import numpy as np
import pandas as pd
from math import sqrt
from scipy import stats

# 1. 関数の定義（名前を run_detailed_analysis に統一）
def run_detailed_analysis(staff_stage1, staff_stage2, n_reps=N_REPS):
    res = {
        'Wait1': np.empty(n_reps),
        'Service1': np.empty(n_reps),
        'Wait2': np.empty(n_reps),
        'Service2': np.empty(n_reps)
    }
    
    for r in range(n_reps):
        rs = np.random.RandomState(SEED + r)
        interarr = rs.exponential(MEAN_INTERARRIVAL, size=N_CUSTOMERS)
        arrivals = np.cumsum(interarr)
        service_B_common = rs.exponential(MEAN_SERVICE_B, size=N_CUSTOMERS)
        
        # サービス時間の割り当て（Aは4秒固定、Bは指数分布）
        s1 = np.full(N_CUSTOMERS, SERVICE_A) if staff_stage1 == 'A' else service_B_common.copy()
        s2 = np.full(N_CUSTOMERS, SERVICE_A) if staff_stage2 == 'A' else service_B_common.copy()
        
        w1, w2 = np.empty(N_CUSTOMERS), np.empty(N_CUSTOMERS)
        end1 = end2 = 0.0
        
        for i in range(N_CUSTOMERS):
            # カウンターⅠ
            start1 = max(arrivals[i], end1)
            w1[i] = start1 - arrivals[i]
            finish1 = start1 + s1[i]
            end1 = finish1
            # カウンターⅡ
            start2 = max(finish1, end2)
            w2[i] = start2 - finish1
            finish2 = start2 + s2[i]
            end2 = finish2
            
        res['Wait1'][r] = w1.mean()
        res['Service1'][r] = s1.mean()
        res['Wait2'][r] = w2.mean()
        res['Service2'][r] = s2.mean()
    return res

# 2. シミュレーション実行
detailed_results = {
    'Cond I (A-B)': run_detailed_analysis('A', 'B'),
    'Cond II (B-A)': run_detailed_analysis('B', 'A')
}

# 3. 統計計算とテーブル作成
final_rows = []
metrics = ['Wait1', 'Service1', 'Wait2', 'Service2']

for m in metrics:
    data_I = detailed_results['Cond I (A-B)'][m]
    data_II = detailed_results['Cond II (B-A)'][m]
    
    t_stat, p_val = stats.ttest_rel(data_II, data_I)
    star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    
    for label, data in [('Cond I (A-B)', data_I), ('Cond II (B-A)', data_II)]:
        mean = np.mean(data)
        se = np.std(data, ddof=1) / sqrt(N_REPS)
        ci_low, ci_up = stats.t.interval(0.95, N_REPS-1, loc=mean, scale=se)
        
        final_rows.append({
            'Metric': m,
            'Config': label,
            'Mean': mean,
            'SE': se,
            '95%CI_Low': ci_low,
            '95%CI_Up': ci_up,
            'Max': np.max(data),
            'Min': np.min(data),
            'Sig': star if label == 'Cond II (B-A)' else ""
        })

df_final = pd.DataFrame(final_rows)
print("\n=== Detailed Process Analysis ===")
print(df_final.to_string(index=False, float_format=lambda x: f"{x:.4f}"))