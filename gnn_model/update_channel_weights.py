import pandas as pd
import yaml


def compute_channel_weights(
    metrics_path, output_yaml, rmse_output_csv, instrument_id=0, use_best_epoch=True
):
    # Load metrics.csv
    df = pd.read_csv(metrics_path)

    # Select row
    if use_best_epoch and "val_loss" in df.columns:
        selected_row = df.loc[df["val_loss"].idxmin()]
    else:
        selected_row = df[df["epoch"] == df["epoch"].max()].iloc[0]

    # Extract val_rmse_ch_* columns
    rmse_cols = [col for col in df.columns if col.startswith("val_rmse_ch_")]
    channel_rmse = selected_row[rmse_cols]

    # Save raw RMSE values
    channel_rmse.to_csv(rmse_output_csv, index=True)
    print(f"Saved per-channel RMSE to: {rmse_output_csv}")

    # Compute inverse RMSE weights and normalize
    epsilon = 1e-6
    raw_weights = 1 / (channel_rmse + epsilon)
    normalized_weights = raw_weights / raw_weights.sum()
    weight_list = [round(w, 6) for w in normalized_weights]

    # Save to YAML
    weights_config = {"channel_weights": {instrument_id: weight_list}}
    with open(output_yaml, "w") as f:
        yaml.dump(weights_config, f)
    print(f"Saved normalized channel weights to: {output_yaml}")


# Example usage
if __name__ == "__main__":
    compute_channel_weights(
        metrics_path="logs/ocelot_gnn/version_0/metrics.csv",
        output_yaml="configs/weights_config.yaml",
        rmse_output_csv="logs/ocelot_gnn/version_0/channel_rmse.csv",
        instrument_id=0,
    )
