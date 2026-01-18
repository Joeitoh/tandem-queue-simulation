import numpy as np
import pandas as pd
from math import sqrt
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import trange

#==============================================================================
# 総系内滞在時間の比較（タンデム待ち行列モデル）
#==============================================================================

# サービスモデルを関数化（スタッフを表す文字列 'A' or 'B'）
def service_time(staff, n):
    if staff == 'A':
        return np.full(n, SERVICE_A)
    elif staff == 'B':
        return np.random.exponential(MEAN_SERVICE_B, size=n)
    else:
        raise ValueError("staff must be 'A' or 'B'")

# タンデム待ち行列をシミュレートして「各客の到着からステージ2完了までの時間」を返す
def simulate_one_run(staff_stage1, staff_stage2, n_customers=N_CUSTOMERS, mean_interarrival=MEAN_INTERARRIVAL):
    # 到着時刻列
    interarr = np.random.exponential(mean_interarrival, size=n_customers)
    arrivals = np.cumsum(interarr)
    # サービス時間列（スタッフB のサンプルを両配置で同じにするため外で用意する）
    # しかしここは1回分だけのシミュレートなので内部で生成してよい
    s1 = service_time(staff_stage1, n_customers)
    s2 = service_time(staff_stage2, n_customers)
    end1 = 0.0
    end2 = 0.0
    finish2_times = np.empty(n_customers)
    for i in range(n_customers):
        start1 = max(arrivals[i], end1)
        finish1 = start1 + s1[i]
        end1 = finish1
        start2 = max(finish1, end2)
        finish2 = start2 + s2[i]
        end2 = finish2
        finish2_times[i] = finish2
    return finish2_times - arrivals  # 各客の総滞在時間



#==============================================================================
# 分布の確認：レプリケーションして「1日あたりの平均滞在時間」の分布を返す
#==============================================================================
def replicate_means(staff_stage1, staff_stage2, n_reps=N_REPS):
    means = np.empty(n_reps)
    # 同じ乱数列でペア化するため、先に全レプリケーション用の乱数状態を保存しておく
    rng_states = [np.random.RandomState(SEED + r) for r in range(n_reps)]
    for r in range(n_reps):
        rs = rng_states[r]
        # ここでは到着列と "B のサービス乱数" を同じにするため、
        # RNG を直接使ってデータを生成（外部から乱数を固定すればペア比較が容易）
        interarr = rs.exponential(MEAN_INTERARRIVAL, size=N_CUSTOMERS)
        arrivals = np.cumsum(interarr)
        # B のサービス列はここで共通に生成し、配置によって割り当てる
        service_B_common = rs.exponential(MEAN_SERVICE_B, size=N_CUSTOMERS)
        # stage1, stage2 のサービス列を決める（A は定数）
        if staff_stage1 == 'A':
            s1 = np.full(N_CUSTOMERS, SERVICE_A)
        else:  # B
            s1 = service_B_common.copy()
        if staff_stage2 == 'A':
            s2 = np.full(N_CUSTOMERS, SERVICE_A)
        else:
            s2 = service_B_common.copy()
        # simulate
        end1 = end2 = 0.0
        finish2_times = np.empty(N_CUSTOMERS)
        for i in range(N_CUSTOMERS):
            start1 = max(arrivals[i], end1)
            finish1 = start1 + s1[i]
            end1 = finish1
            start2 = max(finish1, end2)
            finish2 = start2 + s2[i]
            end2 = finish2
            finish2_times[i] = finish2
        means[r] = (finish2_times - arrivals).mean()
    return means

# *******************************************************************************
# 1.レプリケーション実行（同一乱数列でペア比較できるように、replicate_means 内で調整）***************************
for config in configs:
    print(f"Running reps for config: {config}")
    means = replicate_means(config[0], config[1], n_reps=N_REPS)
    
    m = means.mean()
    se = means.std(ddof=1) / sqrt(N_REPS)
    mx = means.max()
    mn = means.min()
    ci = (m - 1.96 * se, m + 1.96 * se)
    
    # ここで全てのキーを正しく保存します
    results[config] = {
        'means': means,
        'mean': m,
        'se': se,     
        'max': mx,     
        'min': mn,      
        'ci': ci
    }
    print(f"  mean = {m:.4f}s, 95%CI = [{ci[0]:.4f}, {ci[1]:.4f}], SE = {se:.4f}, Max = {mx:.4f}, Min = {mn:.4f}")

# --- 2. 結果表として整形する部分 ---
def sig_star_p(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

# 結果表として整形
rows = []
star = sig_star_p(pval) # p値からスターを取得

for config, v in results.items():
    # v['se'] などが上記で保存されているので、KeyError は出なくなります
    rows.append({
        'Config': f"{config[0]} then {config[1]}",
        'Mean [s]': v['mean'],
        'SE': v['se'],
        '95%CI_Low': v['ci'][0],
        '95%CI_Up': v['ci'][1],
        'Max': v['max'],
        'Min': v['min'],
        'Sig': star
    })

df = pd.DataFrame(rows)
print("\nSummary:")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# 対応の t 検定（差の有意性）*************************
m1 = results[('A','B')]['means']   # A then B
m2 = results[('B','A')]['means']   # B then A

diff = m2 - m1                     # 差（正なら B→A の方が遅い）
N = len(diff)

# 基本統計量
mean_diff = diff.mean()
sd_diff   = diff.std(ddof=1)
se_diff   = sd_diff / sqrt(N)
max_diff  = diff.max()
min_diff  = diff.min()

# 95% 信頼区間（対応 t）
t_crit = stats.t.ppf(0.975, df=N-1)
ci_low = mean_diff - t_crit * se_diff
ci_up  = mean_diff + t_crit * se_diff

# t 値・p 値
t_stat = mean_diff / se_diff
pval = 2 * (1 - stats.t.cdf(abs(t_stat), df=N-1))

# 出力
print(
    f"\nPaired diff (B→A − A→B) | "
    f"Mean={mean_diff:.6f}s | "
    f"SE={se_diff:.6f} | "
    f"95%CI=[{ci_low:.6f}, {ci_up:.6f}] | "
    f"Max={max_diff:.6f} | "
    f"Min={min_diff:.6f} | "
    f"t({N-1})={t_stat:.4f} | "
    f"p={pval:.3e}"
    f" {star}")

# 結果表として整形
rows = []
for config, v in results.items():
    rows.append({
        'Config': f"{config[0]} then {config[1]}",
        'Mean [s]': v['mean'],
        'SE': v['se'],                # 標準誤差を追加
        '95%CI_Low': v['ci'][0],
        '95%CI_Up': v['ci'][1],
        'Max': v['max'],              # 最大値を追加
        'Min': v['min']               # 最小値を追加
    })

df = pd.DataFrame(rows)

print("\n=== Summary Table ===")
# 小数点以下の桁数を揃えて表示
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# 最後に p値を表示して、この表の差が有意であることを示す
print(f"\nStatistical Significance (p-value): {pval:.3e}")

# *******************************************************************************
### ヒストグラム描画
# 1. 共通のビン（区切り）を作成する
all_data = np.concatenate([results[('A','B')]['means'], results[('B','A')]['means']])
# 全体の最小から最大までを均等に30分割する
common_bins = np.linspace(all_data.min(), all_data.max(), 31)

plt.figure(figsize=(10, 6))

# 2. 共通のビンを使って描画
plt.hist(results[('A','B')]['means'], bins=common_bins, alpha=0.5, 
         label='A then B', color='skyblue', edgecolor='white')
plt.hist(results[('B','A')]['means'], bins=common_bins, alpha=0.5, 
         label='B then A', color='orange', edgecolor='white')

# 3. 平均値を垂直線で表示（オプション：より分かりやすくなります）
plt.axvline(results[('A','B')]['mean'], color='blue', linestyle='dashed', linewidth=1.5, label='Mean (A then B)')
plt.axvline(results[('B','A')]['mean'], color='red', linestyle='dashed', linewidth=1.5, label='Mean (B then A)')

# グラフの装飾
plt.title("Distribution of Mean Total Time Spent in the System", fontsize=14)
plt.xlabel("Average Stay Time (seconds)", fontsize=12)
plt.ylabel("Frequency (Number of Replications)", fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.show()

#********************************************************************************

# ボックスプロット描画/ 箱ひげ図をプロット
# データの準備
data_to_plot = [results[('A', 'B')]['means'], results[('B', 'A')]['means']]
labels = ['A then B', 'B then A']

plt.figure(figsize=(8, 6))

# boxplotの描画
bplot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True, medianprops={'color': 'black'})

# 色付け（A then B を薄青、B then A を薄オレンジに）
colors = ['#add8e6', '#ffcc99']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.title("Comparison of System Residence Time")
plt.ylabel("Average Stay Time (seconds)")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# ここで、外れ値が発生した。この原因を探る。******************
# まず外れ値の発生件数を計算

def count_outliers(data):
    # 四分位範囲 (IQR) を計算
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    # 外れ値の境界（上側のみ）を計算
    upper_bound = q3 + 1.5 * iqr
    # 境界を超えるデータの件数をカウント
    outliers = data[data > upper_bound]
    return len(outliers), upper_bound

print("=== Outlier Analysis (Worst Cases) ===")
for config in configs:
    means = results[config]['means']
    count, threshold = count_outliers(means)
    percentage = (count / N_REPS) * 100
    
    config_name = f"{config[0]} then {config[1]}"
    print(f"Config: {config_name}")
    print(f"  Threshold (Upper): {threshold:.4f}s")
    print(f"  Count of Outliers: {count} / {N_REPS} reps")
    print(f"  Risk Rate: {percentage:.2f}%")
    print("-" * 30)
