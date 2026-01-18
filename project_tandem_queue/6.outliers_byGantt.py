
# ここから、外れ値発生の原因を探る（ケース２）    
import matplotlib.pyplot as plt
import numpy as np

# --- 1. ワーストレプリケーション（最も待ちがひどかった回）の特定 ---
# 条件Ⅱ (B-A) において、Wait1 と Wait2 の合計平均が最大だったインデックスを探す
total_waits = detailed_results['Cond II (B-A)']['Wait1'] + detailed_results['Cond II (B-A)']['Wait2']
worst_rep_idx = np.argmax(total_waits)
worst_seed = SEED + worst_rep_idx

# その時のシードで、客1人ずつの詳細データを再生成
rs = np.random.RandomState(worst_seed)
interarr = rs.exponential(MEAN_INTERARRIVAL, size=N_CUSTOMERS)
arrivals = np.cumsum(interarr)
service_B_raw = rs.exponential(MEAN_SERVICE_B, size=N_CUSTOMERS)

# 条件Ⅱ (B-A) の割り当て
s1 = service_B_raw.copy()          # Stage 1: Staff B (指数分布)
s2 = np.full(N_CUSTOMERS, SERVICE_A) # Stage 2: Staff A (固定4秒)

# 1人ずつの開始・終了時刻を計算
start1, finish1 = np.zeros(N_CUSTOMERS), np.zeros(N_CUSTOMERS)
start2, finish2 = np.zeros(N_CUSTOMERS), np.zeros(N_CUSTOMERS)
e1 = e2 = 0.0

for i in range(N_CUSTOMERS):
    # Wait1 & Service1
    start1[i] = max(arrivals[i], e1)
    finish1[i] = start1[i] + s1[i]
    e1 = finish1[i]
    
    # Wait2 & Service2
    start2[i] = max(finish1[i], e2)
    finish2[i] = start2[i] + s2[i]
    e2 = finish2[i]

# --- 2. ガントチャートの描画 ---
plt.figure(figsize=(14, 8))
n_show = 100 # 先頭30名を表示

for i in range(n_show):
    # Wait1: 到着からサービス1開始まで
    plt.plot([arrivals[i], start1[i]], [i, i], color='red', lw=1, ls='--', alpha=0.6)
    # Service1: 窓口1での対応
    plt.hlines(i, start1[i], finish1[i], colors='orange', lw=6, label='Service1 (B)' if i==0 else "")
    
    # Wait2: サービス1終了からサービス2開始まで
    plt.plot([finish1[i], start2[i]], [i, i], color='blue', lw=1, ls='--', alpha=0.6)
    # Service2: 窓口2での対応
    plt.hlines(i, start2[i], finish2[i], colors='skyblue', lw=6, label='Service2 (A)' if i==0 else "")

# 補助情報の描画
plt.scatter(arrivals[:n_show], range(n_show), color='black', marker='v', s=20, label='Arrival', zorder=5)
plt.title(f'Micro-Analysis of Worst Case: Trial #{worst_rep_idx} (Condition II: B-A)', fontsize=14)
plt.xlabel('Time Progress (seconds)')
plt.ylabel('Customer ID')
plt.grid(axis='x', alpha=0.3)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.gca().invert_yaxis() # 客番号を上から順に
plt.tight_layout()
plt.show()

# 3. 原因の数値出力
print(f"--- Analysis of Trial #{worst_rep_idx} ---")
print(f"Max Service1 Time (B): {np.max(s1):.2f}s")
print(f"Avg Wait1 in this trial: {np.mean(start1-arrivals):.2f}s")
#-----------------------------------------------------------------------------------
#ケース１の外れ値発生原因を探る（一応）
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 条件Ⅰのワーストレプリケーションの特定 ---
# Wait1 と Wait2 の合計平均が最大だったインデックスを探す
total_waits_I = detailed_results['Cond I (A-B)']['Wait1'] + detailed_results['Cond I (A-B)']['Wait2']
worst_rep_idx_I = np.argmax(total_waits_I)
worst_seed_I = SEED + worst_rep_idx_I

# その時のシードで、客1人ずつの詳細データを再生成
rs = np.random.RandomState(worst_seed_I)
interarr = rs.exponential(MEAN_INTERARRIVAL, size=N_CUSTOMERS)
arrivals = np.cumsum(interarr)
service_B_raw = rs.exponential(MEAN_SERVICE_B, size=N_CUSTOMERS)

# 条件Ⅰ (A-B) の割り当て
s1 = np.full(N_CUSTOMERS, SERVICE_A) # Stage 1: Staff A (固定4秒)
s2 = service_B_raw.copy()          # Stage 2: Staff B (指数分布)

# 1人ずつの開始・終了時刻を計算
start1, finish1 = np.zeros(N_CUSTOMERS), np.zeros(N_CUSTOMERS)
start2, finish2 = np.zeros(N_CUSTOMERS), np.zeros(N_CUSTOMERS)
e1 = e2 = 0.0

for i in range(N_CUSTOMERS):
    # Stage 1 (A)
    start1[i] = max(arrivals[i], e1)
    finish1[i] = start1[i] + s1[i]
    e1 = finish1[i]
    
    # Stage 2 (B)
    start2[i] = max(finish1[i], e2)
    finish2[i] = start2[i] + s2[i]
    e2 = finish2[i]

# --- 2. ガントチャートの描画 ---
plt.figure(figsize=(14, 8))
n_show = 100 # 先頭30名

for i in range(n_show):
    # Wait1: 赤の点線 (Staff Aを待つ)
    plt.plot([arrivals[i], start1[i]], [i, i], color='red', lw=1, ls='--', alpha=0.6)
    # Service1: 青の太線 (Staff A)
    plt.hlines(i, start1[i], finish1[i], colors='blue', lw=6, label='Service1 (A)' if i==0 else "")
    
    # Wait2: 青の点線 (Staff Bを待つ)
    plt.plot([finish1[i], start2[i]], [i, i], color='blue', lw=1, ls='--', alpha=0.6)
    # Service2: オレンジの太線 (Staff B)
    plt.hlines(i, start2[i], finish2[i], colors='orange', lw=6, label='Service2 (B)' if i==0 else "")

plt.scatter(arrivals[:n_show], range(n_show), color='black', marker='v', s=20, label='Arrival', zorder=5)
plt.title(f'Micro-Analysis of Case I: Trial #{worst_rep_idx_I} (Condition I: A-B)', fontsize=14)
plt.xlabel('Time Progress (seconds)')
plt.ylabel('Customer ID')
plt.grid(axis='x', alpha=0.3)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print(f"Case I - Worst Trial Index: {worst_rep_idx_I}")