import numpy as np
import os

# === 絕對路徑設定（依照你的環境） ===
BASE_DIR = r"C:\Users\chian\OneDrive\桌面\爛新竹\大四\ML_final"

traffic_path = os.path.join(BASE_DIR, "synthetic_PeMSD7_traffic.npy")
timestamps_path = os.path.join(BASE_DIR, "synthetic_PeMSD7_timestamps.npy")
adj_path = os.path.join(BASE_DIR, "synthetic_PeMSD7_adjacency.npy")

print("Loading traffic from:", traffic_path)
traffic = np.load(traffic_path)          # shape: (T, N)
timestamps = np.load(timestamps_path)    # shape: (T,)
adjacency = np.load(adj_path)            # shape: (N, N)

T, N = traffic.shape
print(f"Traffic shape: T={T}, N={N}")
print("Timestamps shape:", timestamps.shape)
print("Adjacency shape:", adjacency.shape)

# === 1. Min-Max Normalization（用 train 部分決定尺度） ===

# 時間序列只沿時間切，假設前 60% 的時間當作 train
train_ratio = 0.6
train_T = int(T * train_ratio)

traffic_train = traffic[:train_T]   # (train_T, N)

# per-sensor min / max
eps = 1e-6
train_min = traffic_train.min(axis=0, keepdims=True)  # (1, N)
train_max = traffic_train.max(axis=0, keepdims=True)  # (1, N)

denom = (train_max - train_min)
denom[denom < eps] = 1.0  # 避免除以 0

traffic_norm = (traffic - train_min) / denom
print("Normalized traffic range (approx):",
      float(traffic_norm.min()), "→", float(traffic_norm.max()))

# === 2. Sliding window 建 X, y ===

past_len = 12    # 過去 12 步 (1 hr)
future_len = 12  # 未來 12 步 (1 hr)

num_samples = T - past_len - future_len + 1
print("Number of sliding window samples:", num_samples)

X = np.zeros((num_samples, past_len, N), dtype=np.float32)
y = np.zeros((num_samples, future_len, N), dtype=np.float32)

for i in range(num_samples):
    X[i] = traffic_norm[i : i + past_len]
    y[i] = traffic_norm[i + past_len : i + past_len + future_len]

print("X shape:", X.shape)  # (num_samples, 12, N)
print("y shape:", y.shape)  # (num_samples, 12, N)

# === 3. 依照時間順序切 train / val / test ===

train_ratio = 0.6
val_ratio = 0.2  # test 會是剩下的 0.2

train_end = int(num_samples * train_ratio)
val_end = int(num_samples * (train_ratio + val_ratio))

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

print("Train size:", X_train.shape[0])
print("Val size:  ", X_val.shape[0])
print("Test size: ", X_test.shape[0])

# === 4. 存成 .npy，之後模型可以直接載入 ===

np.save(os.path.join(BASE_DIR, "X_train.npy"), X_train)
np.save(os.path.join(BASE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(BASE_DIR, "X_val.npy"), X_val)
np.save(os.path.join(BASE_DIR, "y_val.npy"), y_val)
np.save(os.path.join(BASE_DIR, "X_test.npy"), X_test)
np.save(os.path.join(BASE_DIR, "y_test.npy"), y_test)

# 也把 scaler 存起來，之後做反標準化用
np.save(os.path.join(BASE_DIR, "scaler_min.npy"), train_min.astype(np.float32))
np.save(os.path.join(BASE_DIR, "scaler_max.npy"), train_max.astype(np.float32))

print("\nSaved preprocessed files to", BASE_DIR)
print("  X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy")
print("  scaler_min.npy, scaler_max.npy")
