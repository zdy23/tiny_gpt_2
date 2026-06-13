"""从 checkpoints/metrics.csv 绘制训练损失曲线"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("checkpoints/metrics.csv")


def moving_average(x, window=50):
    pad = window // 2
    return np.convolve(np.pad(x, (pad, pad), mode="reflect"), np.ones(window) / window, mode="valid")[:len(x)]


plt.style.use("seaborn-v0_8-darkgrid")
fig, ax = plt.subplots(figsize=(14, 5.5))
fig.patch.set_facecolor("#fafafa")
ax.set_facecolor("#fafafa")

ax.plot(df["step"], df["train_loss"], label="Train Loss (raw)", alpha=0.20, linewidth=0.6, color="#4C72B0")
window = max(1, len(df) // 40)
smooth = moving_average(df["train_loss"].values, window)
ax.plot(df["step"], smooth, label=f"Train Loss (smooth, w={window})", linewidth=1.6, color="#4C72B0")
residuals = df["train_loss"].values - smooth
ax.fill_between(df["step"], smooth - residuals.std(), smooth + residuals.std(),
                color="#4C72B0", alpha=0.08, label="+-1sigma band")

valid_val = df.dropna(subset=["val_loss"])
if len(valid_val) > 0:
    ax.plot(valid_val["step"], valid_val["val_loss"], color="#C44E52", linewidth=1.2,
            marker="o", markersize=5, label="Val Loss", zorder=5)
    best_idx = valid_val["val_loss"].idxmin()
    best_row = df.loc[best_idx]
    ax.scatter(best_row["step"], best_row["val_loss"], color="#C44E52", s=120,
              edgecolors="white", linewidth=1.2, zorder=6)
    ax.annotate(f"Best: {best_row['val_loss']:.4f} @ step {int(best_row['step'])}",
                xy=(best_row["step"], best_row["val_loss"]), xytext=(-60, -40),
                textcoords="offset points", fontsize=9, color="#C44E52", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C44E52", alpha=0.8))

final = df.iloc[-1]
ax.annotate(f"Final: {final['train_loss']:.4f}", xy=(final["step"], final["train_loss"]),
            xytext=(-120, 30), textcoords="offset points", fontsize=9, color="#4C72B0", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#4C72B0", alpha=0.8))

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Tiny GPT-2 Training Loss", fontsize=14, fontweight="bold")
ax.legend(loc="best", framealpha=0.9, edgecolor="#cccccc", fontsize=10, ncol=2)
ax.tick_params(labelsize=10)
ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["bottom", "left"]:
    ax.spines[s].set_color("#cccccc")

fig.tight_layout()
fig.savefig("checkpoints/loss_curve.png", dpi=200, bbox_inches="tight")
print("Saved checkpoints/loss_curve.png")
plt.show()
