import pandas as pd
import matplotlib.pyplot as plt

# Load your predictions CSV
df = pd.read_csv("bt_predictions_epoch0.csv")

# Plot predictions vs targets for each channel
for i in range(1, 23):  # Channels 1 to 22
    plt.figure(figsize=(12, 5))

    # Plot target
    plt.subplot(1, 2, 1)
    sc1 = plt.scatter(df["lon_deg"], df["lat_deg"], c=df[f"true_bt_{i}"], cmap="viridis", s=5)
    plt.colorbar(sc1)
    plt.title(f"True Brightness Temp - Channel {i}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Plot prediction
    plt.subplot(1, 2, 2)
    sc2 = plt.scatter(df["lon_deg"], df["lat_deg"], c=df[f"pred_bt_{i}"], cmap="viridis", s=5)
    plt.colorbar(sc2)
    plt.title(f"Predicted Brightness Temp - Channel {i}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(f"bt_map_channel_{i}.png")
    plt.close()
