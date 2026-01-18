# *******************************************
# スタッフの稼働率
# ********************************************
def calculate_utilization(detailed_res, n_reps=500):
    util_stage1 = np.empty(n_reps)
    util_stage2 = np.empty(n_reps)
    
    for r in range(n_reps):
        # サービスの合計時間を算出（Service1, Service2 の平均 * 客数）
        # シミュレーションの終了時刻は、最後の客がStage2を終えた時刻
        # ここでは簡略化のため、「最後の客の終了時刻 - 最初の客の到着時刻」を全稼働時間とする
        
        # 本来の稼働率 = (接客時間の総計) / (全シミュレーション時間)
        # 1レプリケーションあたりの接客時間の総和
        total_service1 = detailed_res['Service1'][r] * N_CUSTOMERS
        total_service2 = detailed_res['Service2'][r] * N_CUSTOMERS
        
        # 全体時間（最後の客の退出時刻を近似的に算出）
        # 厳密には各レプリケーションの終了時刻が必要なため、平均的なスループットで近似
        total_time = N_CUSTOMERS * MEAN_INTERARRIVAL 
        
        util_stage1[r] = total_service1 / total_time
        util_stage2[r] = total_service2 / total_time
        
    return util_stage1.mean(), util_stage2.mean()

# 計算実行
u1_I, u2_I = calculate_utilization(detailed_results['Cond I (A-B)'])
u1_II, u2_II = calculate_utilization(detailed_results['Cond II (B-A)'])

print(f"=== Staff Utilization (Average) ===")
print(f"Cond I (A-B)  | Stage1(A): {u1_I*100:.2f}% | Stage2(B): {u2_I*100:.2f}%")
print(f"Cond II (B-A) | Stage1(B): {u1_II*100:.2f}% | Stage2(A): {u2_II*100:.2f}%")


#****************************************************************
##ケース１のスタッフ稼働率と行列長の詳細図
#****************************************************************
import matplotlib.pyplot as plt
import numpy as np

from tandem_queue.config import MEAN_INTERARRIVAL, MEAN_SERVICE_B, N_CUSTOMERS, N_REPS, SEED, N_REPS, SERVICE_A

# --- 1. 時系列データの準備（6000秒間） ---
max_t = 2000
t_axis = np.arange(max_t)

# 窓関数を「4秒」に設定
# これにより、Staff Aの「4秒固定」の動きとBのバラツキが線として見えてきます
window = 2 

# 稼働フラグの計算
timeline_A = np.zeros(max_t)
timeline_B = np.zeros(max_t)
for i in range(N_CUSTOMERS):
    if start1[i] < max_t:
        timeline_A[int(start1[i]):int(min(finish1[i], max_t))] = 1
    if start2[i] < max_t:
        timeline_B[int(start2[i]):int(min(finish2[i], max_t))] = 1

util_A_smooth = np.convolve(timeline_A, np.ones(window)/window, mode='same')
util_B_smooth = np.convolve(timeline_B, np.ones(window)/window, mode='same')

# 行列の長さを計算
lq1_ts = np.zeros(max_t)
lq2_ts = np.zeros(max_t)
for t in range(max_t):
    lq1_ts[t] = np.sum((arrivals <= t) & (start1 > t))
    lq2_ts[t] = np.sum((finish1 <= t) & (start2 > t))

# --- 2. グラフ作成 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# 上段: スタッフ稼働率
ax1.plot(t_axis, util_A_smooth[:max_t], label='Staff A (Stage 1) Util', color='blue', lw=1, alpha=0.8)
ax1.plot(t_axis, util_B_smooth[:max_t], label='Staff B (Stage 2) Util', color='orange', lw=1, alpha=0.6)

# Efficiency Loss: A(前)が動いているのにB(後)が暇をしている領域
ax1.fill_between(t_axis, util_B_smooth[:max_t], util_A_smooth[:max_t], 
                 where=(util_A_smooth[:max_t] > util_B_smooth[:max_t]),
                 color='red', alpha=0.3, label='Efficiency Loss')

ax1.set_ylabel('Utilization Rate (2s Moving Average)')
ax1.set_ylim(0, 1.1)
ax1.set_title('Condition I (A-B): 2000s Analysis (2s Window)', fontsize=14)
ax1.legend(loc='lower right', fontsize='small', ncol=3)
ax1.grid(True, alpha=0.2)

# 下段: 行列の長さ
ax2.fill_between(t_axis, lq1_ts, color='blue', alpha=0.3, label='Wait1 (Gate)')
ax2.plot(t_axis, lq1_ts, color='blue', lw=0.5)
ax2.fill_between(t_axis, lq2_ts, color='orange', alpha=0.3, label='Wait2 (Middle)')
ax2.plot(t_axis, lq2_ts, color='orange', lw=0.5)

ax2.set_ylabel('Number of People Waiting')
ax2.set_xlabel('Time (seconds)')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()



#****************************************************************
##ケース２のスタッフ稼働率と行列長の詳細図
#****************************************************************
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 時系列データの準備（2000秒間） ---
max_t = 2000
t_axis = np.arange(max_t)

# 窓関数を「2秒」に設定
window = 2 

# 稼働フラグの計算（Case II の結果を参照）
# Stage 1 = Staff B (バラツキあり), Stage 2 = Staff A (固定4秒)
timeline_B = np.zeros(max_t) # Stage 1
timeline_A = np.zeros(max_t) # Stage 2

for i in range(N_CUSTOMERS):
    if start1[i] < max_t:
        timeline_B[int(start1[i]):int(min(finish1[i], max_t))] = 1
    if start2[i] < max_t:
        timeline_A[int(start2[i]):int(min(finish2[i], max_t))] = 1

# 移動平均による平滑化
util_B_smooth = np.convolve(timeline_B, np.ones(window)/window, mode='same')
util_A_smooth = np.convolve(timeline_A, np.ones(window)/window, mode='same')

# 行列の長さを計算
lq1_ts = np.zeros(max_t)
lq2_ts = np.zeros(max_t)
for t in range(max_t):
    # Wait 1: Staff B の前
    lq1_ts[t] = np.sum((arrivals <= t) & (start1 > t))
    # Wait 2: Staff A の前
    lq2_ts[t] = np.sum((finish1 <= t) & (start2 > t))

# --- 2. グラフ作成 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# 上段: スタッフ稼働率
# Bが前段、Aが後段
ax1.plot(t_axis, util_B_smooth[:max_t], label='Staff B (Stage 1) Util', color='orange', lw=1, alpha=0.8)
ax1.plot(t_axis, util_A_smooth[:max_t], label='Staff A (Stage 2) Util', color='blue', lw=1, alpha=0.6)

# Efficiency Loss: 前段(B)が稼働しているのに、後段(A)に客が回っていない隙間を赤く塗る
ax1.fill_between(t_axis, util_A_smooth[:max_t], util_B_smooth[:max_t], 
                 where=(util_B_smooth[:max_t] > util_A_smooth[:max_t]),
                 color='red', alpha=0.3, label='Efficiency Loss')

ax1.set_ylabel('Utilization Rate (2s Moving Average)')
ax1.set_ylim(0, 1.1)
ax1.set_title('Condition II (B-A): 2000s Analysis (2s Window)', fontsize=14)
ax1.legend(loc='lower right', fontsize='small', ncol=3)
ax1.grid(True, alpha=0.2)

# 下段: 行列の長さ
# Bが前なので Wait1=オレンジ、Aが後ろなので Wait2=青
ax2.fill_between(t_axis, lq1_ts, color='orange', alpha=0.3, label='Wait1 (Gate: B)')
ax2.plot(t_axis, lq1_ts, color='orange', lw=0.5)
ax2.fill_between(t_axis, lq2_ts, color='blue', alpha=0.3, label='Wait2 (Middle: A)')
ax2.plot(t_axis, lq2_ts, color='blue', lw=0.5)

ax2.set_ylabel('Number of People Waiting')
ax2.set_xlabel('Time (seconds)')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()



#****************************************************************
# 勤怠管理視点での遊休時間算出
#****************************************************************
def replicate_idle_ratios(staff_stage1, staff_stage2, n_reps=N_REPS):
    # 各レプリケーションの結果を格納
    idle_ratios_1 = np.empty(n_reps)
    idle_ratios_2 = np.empty(n_reps)
    
    rng_states = [np.random.RandomState(SEED + r) for r in range(n_reps)]
    
    for r in range(n_reps):
        rs = rng_states[r]
        interarr = rs.exponential(MEAN_INTERARRIVAL, size=N_CUSTOMERS)
        arrivals = np.cumsum(interarr)
        service_B_common = rs.exponential(MEAN_SERVICE_B, size=N_CUSTOMERS)
        
        # スタッフ配置に応じたサービス時間設定
        s1 = np.full(N_CUSTOMERS, SERVICE_A) if staff_stage1 == 'A' else service_B_common
        s2 = np.full(N_CUSTOMERS, SERVICE_A) if staff_stage2 == 'A' else service_B_common
        
        end1 = end2 = 0.0
        total_idle1 = 0.0
        total_idle2 = 0.0
        
        for i in range(N_CUSTOMERS):
            # --- Stage 1 (スタッフ1の遊休計算) ---
            # 客が到着した時に、まだ前の客の対応中なら遊休は0。
            # 前の客が終わってから到着するまでの間が遊休。
            if arrivals[i] > end1:
                total_idle1 += (arrivals[i] - end1)
            
            start1 = max(arrivals[i], end1)
            finish1 = start1 + s1[i]
            end1 = finish1
            
            # --- Stage 2 (スタッフ2の遊休計算) ---
            # Stage 1を終えた客が来た時に、まだ前の客を対応中なら遊休は0。
            # 前の客が終わってからStage 1の客が流れてくるまでが遊休。
            if finish1 > end2:
                total_idle2 += (finish1 - end2)
            
            start2 = max(finish1, end2)
            finish2 = start2 + s2[i]
            end2 = finish2
        
        # シミュレーション全体の終了時間（最後の客の退去時刻）
        total_time = end2
        
        idle_ratios_1[r] = (total_idle1 / total_time) * 100
        idle_ratios_2[r] = (total_idle2 / total_time) * 100
        
    return idle_ratios_1, idle_ratios_2

# 実行例
idle_A_in_AB, idle_B_in_AB = replicate_idle_ratios('A', 'B')
idle_B_in_BA, idle_A_in_BA = replicate_idle_ratios('B', 'A')

print(f"配置 [A -> B] のとき: スタッフA遊休 {idle_A_in_AB.mean():.2f}%, スタッフB遊休 {idle_B_in_AB.mean():.2f}%")
print(f"配置 [B -> A] のとき: スタッフB遊休 {idle_B_in_BA.mean():.2f}%, スタッフA遊休 {idle_A_in_BA.mean():.2f}%")