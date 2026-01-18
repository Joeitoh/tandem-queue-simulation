#****************************************************************
#箱ひげ図で待ち時間の分布を確認する
#****************************************************************
#アウトライヤー確認
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. データの整形（秒単位の待ち時間データを使用）
wait_data = []
for metric in ['Wait1', 'Wait2']:
    for label in ['Cond I (A-B)', 'Cond II (B-A)']:
        values = detailed_results[label][metric]
        for v in values:
            wait_data.append({
                'Location': metric,
                'Condition': label,
                'Wait Time (s)': v
            })

df_wait_plot = pd.DataFrame(wait_data)

# 2. 描画
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# 箱ひげ図の作成
ax = sns.boxplot(
    x='Location', 
    y='Wait Time (s)', 
    hue='Condition', 
    data=df_wait_plot,
    palette=['#3498db', '#e67e22'],
    showmeans=True,
    meanprops={"marker":"D","markerfacecolor":"white", "markeredgecolor":"black"}
)

# 3. 凡例を枠外に配置
#plt.legend(title='Condition', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# 4. 装飾と統計記号（***）の追加
plt.title('Distribution of Waiting Times (N=500)', fontsize=15)
plt.ylabel('Average Waiting Time per Trial (seconds)')

plt.tight_layout()
plt.show()
#外れ値の件数をカウントする関数
import numpy as np

def count_outliers(data):
    # 第1四分位数(25%)と第3四分位数(75%)を算出
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    # 外れ値の境界線を設定
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # 境界線を超えるデータの数をカウント
    return ((data < lower_bound) | (data > upper_bound)).sum()

print("=== 待ち時間（秒）の外れ値発生回数 (N=500) ===")
for m in ['Wait1', 'Wait2']:
    n_I  = count_outliers(detailed_results['Cond I (A-B)'][m])
    n_II = count_outliers(detailed_results['Cond II (B-A)'][m])
    print(f"{m:5} | 条件Ⅰ(A-B): {n_I:2}回, 条件Ⅱ(B-A): {n_II:2}回")


#****************************************************************
#滞在時間の内訳を積み上げ棒グラフで表示
#****************************************************************
import matplotlib.pyplot as plt
import pandas as pd

# 1. データの準備（順序を逆転させると、グラフの上から順にCond I, Cond IIと並びます）
data = {
    'Condition': ['Cond II (B-A)', 'Cond I (A-B)'],
    'Wait1': [16.066, 8.035],
    'Service1': [3.999, 4.000],
    'Wait2': [8.014, 12.042],
    'Service2': [4.000, 3.999]
}
df_plot = pd.DataFrame(data).set_index('Condition')

# 2. 描画設定（kind='barh' で横棒に変更）
fig, ax = plt.subplots(figsize=(12, 6))
df_plot.plot(kind='barh', stacked=True, ax=ax, 
             color=['#3498db', '#e74c3c', '#2ecc71', '#f1c40f'], alpha=0.8)

# 3. 各パーツの中央に数値を表示
for n, (label, _content) in enumerate(df_plot.iterrows()):
    cumulative_width = 0
    for col in df_plot.columns:
        width = df_plot.loc[label, col]
        ax.text(cumulative_width + width/2, n, f'{width:.2f}s', 
                ha='center', va='center', fontweight='bold')
        cumulative_width += width

# 4. 合計時間の表示（棒の右端に表示）
for i, total in enumerate(df_plot.sum(axis=1)):
    ax.text(total + 0.5, i, f'Total: {total:.2f}s', va='center', fontsize=12, fontweight='bold')

# 5. グラフの装飾
ax.set_title('Breakdown of Residence Time: Cond I vs Cond II', fontsize=14)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Configuration')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Metrics')

plt.tight_layout()
plt.show()