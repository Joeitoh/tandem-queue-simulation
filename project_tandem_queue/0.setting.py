import numpy as np
import pandas as pd
from math import sqrt
from scipy import stats
from itertools import permutations
from tqdm import trange
import matplotlib.pyplot as plt

# ---------- パラメータ ----------
SEED = 2025
N_CUSTOMERS = 10000      # 1レプリケーションあたりの客数（問題設定通り）
MEAN_INTERARRIVAL = 5.0  # 到着間隔の平均（秒）
MEAN_SERVICE_B = 4.0     # スタッフBの平均サービス時間（指数分布）
SERVICE_A = 4.0          # スタッフAのサービス時間（定数）
N_REPS = 500             # レプリケーション数（CI推定用）. 必要に応じて増やす
# -------------------------------

np.random.seed(SEED)

# ============================================================
## 1: 実験条件（スタッフ配置）の定義
# ============================================================
# 全探索（ここでは2段しかないので配置は ('A','B') と ('B','A') の2通り）
configs = [('A','B'), ('B','A')]
results = {}
