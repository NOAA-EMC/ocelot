import pandas as pd
import matplotlib.pyplot as plt

# Load your predictions CSV
df = pd.read_csv("bt_predictions_epoch9.csv")

# Plot predictions vs targets for each channel
for i in range(1, 23):  # Channels 1 to 22
    plt.figure(figsize=(16, 5))
    # Get shared color scale limits
    vmin = min(df[f"true_bt_{i}"].min(), df[f"pred_bt_{i}"].min())
    vmax = max(df[f"true_bt_{i}"].max(), df[f"pred_bt_{i}"].max())

    # Plot target
    plt.subplot(1, 3, 1)
    sc1 = plt.scatter(df["lon_deg"], df["lat_deg"], c=df[f"true_bt_{i}"], cmap="viridis", s=5, vmin=vmin, vmax=vmax)
    plt.colorbar(sc1)
    plt.title(f"True Brightness Temp - Channel {i}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Plot prediction
    plt.subplot(1, 3, 2)
    sc2 = plt.scatter(df["lon_deg"], df["lat_deg"], c=df[f"pred_bt_{i}"], cmap="viridis", s=5, vmin=vmin, vmax=vmax)
    plt.colorbar(sc2)
    plt.title(f"Predicted Brightness Temp - Channel {i}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.subplot(1, 3, 3)
    sc2 = plt.scatter(df["lon_deg"], df["lat_deg"], c=((df[f"true_bt_{i}"]-df[f"pred_bt_{i}"])/df[f"true_bt_{i}"])*100, cmap="bwr", s=5)
    plt.colorbar(sc2)
    plt.title(f"Error Brightness Temp - Channel {i}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(f"bt_map_channel_{i}.png")
    plt.close()
    print(f"finished ch_{i}")
