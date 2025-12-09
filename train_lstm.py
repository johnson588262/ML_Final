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

# ======== LSTM 模型定義 ========

class LSTMForecast(nn.Module):
    """
    很簡單的 encoder-only LSTM：
    輸入: (B, past_len, N)
    LSTM 最後一個 hidden state -> Linear -> (future_len * N)
    """
    def __init__(self, past_len, num_sensors, future_len,
                 hidden_size=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.past_len = past_len
        self.num_sensors = num_sensors
        self.future_len = future_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_dim = future_len * num_sensors
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        # x: (B, past_len, N)
        B = x.size(0)
        lstm_out, (h_n, c_n) = self.lstm(x)  # h_n: (num_layers, B, H)
        h_last = h_n[-1]                     # (B, H)
        y_flat = self.fc(h_last)             # (B, future_len * N)
        y = y_flat.view(B, self.future_len, self.num_sensors)
        return y

# ======== 訓練 & 評估 ========

def train_model(
    X_train, y_train, X_val, y_val,
    past_len=12, future_len=12,
    batch_size=64, num_epochs=50, lr=1e-3,
    hidden_size=128, num_layers=1
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_sensors = X_train.shape[2]

    def to_tensor(x):
        return torch.from_numpy(x).float()

    X_train_t = to_tensor(X_train).to(device)
    y_train_t = to_tensor(y_train).to(device)
    X_val_t = to_tensor(X_val).to(device)
    y_val_t = to_tensor(y_val).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMForecast(
        past_len=past_len,
        num_sensors=num_sensors,
        future_len=future_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.1,
    ).to(device)

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
    model_path = os.path.join(BASE_DIR, "lstm_forecast.pth")
    torch.save(model.state_dict(), model_path)
    print("Saved LSTM model to:", model_path)

    # 畫 loss curve
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.title("LSTM training / validation loss")
    fig_path = os.path.join(BASE_DIR, "lstm_loss_curve.png")
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
        y_pred_t = model(X_test_t)

    y_pred = y_pred_t.cpu().numpy()
    y_true = y_test

    diff = y_pred - y_true
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    print(f"[LSTM] Test RMSE: {rmse:.6f}")
    print(f"[LSTM] Test MAE:  {mae:.6f}")

    # 畫一個 sample + 一個 sensor 的曲線
    sample_idx = 0
    gt_series = y_true[sample_idx, :, sensor_idx]
    pred_series = y_pred[sample_idx, :, sensor_idx]

    plt.figure()
    plt.plot(range(future_len), gt_series, marker="o", label="Ground truth")
    plt.plot(range(future_len), pred_series, marker="x", label="Prediction")
    plt.xlabel("Future time step (5-min intervals)")
    plt.ylabel("Normalized traffic")
    plt.title(f"LSTM forecast - sensor {sensor_idx} (one test sample)")
    plt.legend()
    fig_path = os.path.join(BASE_DIR, f"lstm_prediction_sensor{sensor_idx}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved LSTM prediction plot to:", fig_path)

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
        num_epochs=80,        # 多跑一點
        lr=5e-4,              # lr 調小
        hidden_size=256,      # 更大的 hidden
        num_layers=2          # 兩層 LSTM
    )

    evaluate_and_plot(
        model, X_test, y_test,
        past_len=past_len,
        future_len=future_len,
        sensor_idx=0
    )

if __name__ == "__main__":
    main()
