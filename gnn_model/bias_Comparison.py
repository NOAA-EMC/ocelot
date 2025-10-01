import pandas as pd
import matplotlib.pyplot as plt


def load_rmse_from_metrics(metrics_csv):
    df = pd.read_csv(metrics_csv)

    # Find the last row with valid RMSE values
    rmse_cols = [col for col in df.columns if col.startswith("val_rmse_ch_")]
    df_valid = df.dropna(subset=rmse_cols)

    if df_valid.empty:
        raise ValueError("No valid RMSE values found in metrics file.")

    return df_valid[rmse_cols].iloc[-1].to_dict()


# Load both versions
rmse_v0 = load_rmse_from_metrics("logs/ocelot_gnn/version_0/metrics.csv")
rmse_v1 = load_rmse_from_metrics("logs/ocelot_gnn/version_1/metrics.csv")

# Align by channel
channels = sorted(rmse_v0.keys(), key=lambda x: int(x.split("_")[-1]))
v0_vals = [rmse_v0[ch] for ch in channels]
v1_vals = [rmse_v1[ch] for ch in channels]

for ch in channels:
    delta = rmse_v1[ch] - rmse_v0[ch]
    print(f"{ch}: ΔRMSE = {delta:.2f} ({100 * delta / rmse_v0[ch]:+.1f}%)")


# Plot
plt.figure(figsize=(14, 6))
x = range(len(channels))
plt.plot(x, v0_vals, marker="o", label="Version 0 (Equal Weights)")
plt.plot(x, v1_vals, marker="s", label="Version 1 (Weighted)")
plt.xticks(x, channels, rotation=90)
plt.ylabel("Validation RMSE")
plt.title("Per-Channel RMSE Comparison")
plt.legend()
plt.grid(True, axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("Comparison.png")
