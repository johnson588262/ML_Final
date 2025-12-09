import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ======== 路徑設定 ========
BASE_DIR = r"C:\Users\chian\OneDrive\桌面\爛新竹\大四\ML_final"

def load_data():
    X_train = np.load(os.path.join(BASE_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(BASE_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(BASE_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(BASE_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(BASE_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(BASE_DIR, "y_test.npy"))

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test

# ======== MLP 模型定義（Fully-connected NN） ========

class MLPForecast(nn.Module):
    def __init__(self, past_len, num_sensors, future_len, hidden_dim=1024):
        super().__init__()
        self.past_len = past_len
        self.num_sensors = num_sensors
        self.future_len = future_len

        in_dim = past_len * num_sensors
        out_dim = future_len * num_sensors

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # x: (B, past_len, N)
        B = x.shape[0]
        x_flat = x.view(B, -1)        # (B, past_len * N)
        y_flat = self.net(x_flat)     # (B, future_len * N)
        y = y_flat.view(B, self.future_len, self.num_sensors)
        return y

# ======== 訓練 & 評估 ========

def train_model(
    X_train, y_train, X_val, y_val,
    past_len=12, future_len=12,
    batch_size=64, num_epochs=50, lr=1e-3
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_sensors = X_train.shape[2]
    in_dim = past_len * num_sensors
    out_dim = future_len * num_sensors
    print(f"in_dim={in_dim}, out_dim={out_dim}")

    # numpy -> torch tensors，並 flatten
    def to_tensor(x):
        return torch.from_numpy(x).float()

    X_train_t = to_tensor(X_train).to(device)  # (N_train, past_len, N)
    y_train_t = to_tensor(y_train).to(device)  # (N_train, future_len, N)
    X_val_t = to_tensor(X_val).to(device)
    y_val_t = to_tensor(y_val).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MLPForecast(past_len, num_sensors, future_len, hidden_dim=1024).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)

        avg_train_loss = total_train_loss / len(train_ds)

        # validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                total_val_loss += loss.item() * xb.size(0)
        avg_val_loss = total_val_loss / len(val_ds)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch:03d}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")

    # 存模型
    model_path = os.path.join(BASE_DIR, "mlp_forecast.pth")
    torch.save(model.state_dict(), model_path)
    print("Saved model to:", model_path)

    # 畫 loss curve
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.title("MLP training / validation loss")
    fig_path = os.path.join(BASE_DIR, "mlp_loss_curve.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved loss curve to:", fig_path)

    return model

def evaluate_and_plot(model, X_test, y_test, past_len=12, future_len=12, sensor_idx=0):
    device = next(model.parameters()).device
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    model.eval()
    with torch.no_grad():
        y_pred_t = model(X_test_t)   # (N_test, future_len, N)

    y_pred = y_pred_t.cpu().numpy()
    y_true = y_test

    # 整體指標 (flatten everything)
    diff = y_pred - y_true
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test MAE:  {mae:.6f}")

    # 抽一個 sample + 一個 sensor 畫圖
    sample_idx = 0
    gt_series = y_true[sample_idx, :, sensor_idx]
    pred_series = y_pred[sample_idx, :, sensor_idx]

    plt.figure()
    plt.plot(range(future_len), gt_series, marker="o", label="Ground truth")
    plt.plot(range(future_len), pred_series, marker="x", label="Prediction")
    plt.xlabel("Future time step (5-min intervals)")
    plt.ylabel("Normalized traffic")
    plt.title(f"Sensor {sensor_idx} - one test sample forecast (12-step horizon)")
    plt.legend()
    fig_path = os.path.join(BASE_DIR, f"mlp_prediction_sensor{sensor_idx}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved prediction plot to:", fig_path)

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    past_len = X_train.shape[1]
    future_len = y_train.shape[1]

    model = train_model(
        X_train, y_train,
        X_val, y_val,
        past_len=past_len,
        future_len=future_len,
        batch_size=64,
        num_epochs=50,
        lr=1e-3
    )

    evaluate_and_plot(model, X_test, y_test,
                      past_len=past_len,
                      future_len=future_len,
                      sensor_idx=0)

if __name__ == "__main__":
    main()
