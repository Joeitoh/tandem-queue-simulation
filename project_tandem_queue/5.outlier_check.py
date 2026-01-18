import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# *******************************************************************************
# 箱ひげ図を作成して、外れ値の分布をみる
# *******************************************************************************
# 1. データの整形
data_list = []
for val in res_I['Wait1']:
    data_list.append({'Location': 'Wait1', 'Condition': 'Cond I (A-B)', 'Lq': val})
for val in res_II['Wait1']:
    data_list.append({'Location': 'Wait1', 'Condition': 'Cond II (B-A)', 'Lq': val})
for val in res_I['Wait2']:
    data_list.append({'Location': 'Wait2', 'Condition': 'Cond I (A-B)', 'Lq': val})
for val in res_II['Wait2']:
    data_list.append({'Location': 'Wait2', 'Condition': 'Cond II (B-A)', 'Lq': val})

df_boxplot = pd.DataFrame(data_list)

# 2. 描画
plt.figure(figsize=(11, 6)) # 凡例のスペース分、横幅を少し広げました
sns.set_theme(style="whitegrid")

# 箱ひげ図
ax = sns.boxplot(x='Location', y='Lq', hue='Condition', data=df_boxplot, 
                 palette=['#3498db', '#e67e22'], 
                 showmeans=True,
                 meanprops={"marker":"D","markerfacecolor":"white", "markeredgecolor":"black"})

# --- 凡例の調整 ---
# loc='upper left' (凡例の左上角) を bbox_to_anchor=(1.02, 1) (グラフの枠外右) に固定
#plt.legend(title='Condition', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# 3. 装飾
plt.title('Comparison of Queue Length (Lq) with Outliers', fontsize=14)
plt.ylabel('Average Number of People (Lq)')
plt.xlabel('Location')

plt.tight_layout() # 全体が収まるように自動調整
plt.show()

# *******************************************************************************
#外れ値が出現するので、その出現件数をカウントする関数
def get_outliers_count(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    return ((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).sum()

print("--- Outlier Counts (N=500) ---")
for loc in ['Wait1', 'Wait2']:
    n_I  = get_outliers_count(res_I[loc])
    n_II = get_outliers_count(res_II[loc])
    print(f"{loc} | Cond I: {n_I:>3} cases, Cond II: {n_II:>3} cases")