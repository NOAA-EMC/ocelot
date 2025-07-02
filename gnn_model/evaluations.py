import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Load your predictions CSV
df_bt = pd.read_csv("bt_predictions_epoch12.csv")
df_pressure = pd.read_csv("pressure_predictions_epoch12.csv")

plt.figure(figsize=(12, 5))
# Get shared color scale limits
vmin_pressure = min(df_pressure[f"true_pressure"].min(), df_pressure[f"pred_pressure"].min())
vmax_pressure = max(df_pressure[f"true_pressure"].max(), df_pressure[f"pred_pressure"].max())

# Plot target
plt.subplot(1, 3, 1)
sc1 = plt.scatter(
    df_pressure["lon_deg"], df_pressure["lat_deg"], c=df_pressure["true_pressure"], cmap="viridis", s=5, vmin=vmin_pressure, vmax=vmax_pressure)
plt.colorbar(sc1)
plt.title(f"True Pressure")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Plot prediction
plt.subplot(1, 3, 2)
sc2 = plt.scatter(
    df_pressure["lon_deg"], df_pressure["lat_deg"], c=df_pressure["pred_pressure"], cmap="viridis", s=5, vmin=vmin_pressure, vmax=vmax_pressure)
plt.colorbar(sc2)
plt.title(f"Predicted Pressure")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

error = df_pressure["true_pressure"]-df_pressure["pred_pressure"]
norm = TwoSlopeNorm(vmin=error.min(), vcenter=0, vmax=error.max())
plt.subplot(1, 3, 3)
sc3 = plt.scatter(df_pressure["lon_deg"], df_pressure["lat_deg"], c=error, norm=norm, cmap="bwr", s=5)
plt.colorbar(sc3)
plt.title(f"Error Pressure")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(f"pressure_map.png")
plt.close()
print(f"finished pressure_map.png")

# Plot predictions vs targets for each channel
for i in range(1, 23):  # Channels 1 to 22
    plt.figure(figsize=(16, 5))
    # Get shared color scale limits
    vmin = min(df_bt[f"true_bt_{i}"].min(), df_bt[f"pred_bt_{i}"].min())
    vmax = max(df_bt[f"true_bt_{i}"].max(), df_bt[f"pred_bt_{i}"].max())

    # Plot target
    plt.subplot(1, 3, 1)
    sc1 = plt.scatter(df_bt["lon_deg"], df_bt["lat_deg"], c=df_bt[f"true_bt_{i}"], cmap="viridis", s=5, vmin=vmin, vmax=vmax)
    plt.colorbar(sc1)
    plt.title(f"True Brightness Temp - Channel {i}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Plot prediction
    plt.subplot(1, 3, 2)
    sc2 = plt.scatter(df_bt["lon_deg"], df_bt["lat_deg"], c=df_bt[f"pred_bt_{i}"], cmap="viridis", s=5, vmin=vmin, vmax=vmax)
    plt.colorbar(sc2)
    plt.title(f"Predicted Brightness Temp - Channel {i}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    error = ((df_bt[f"true_bt_{i}"] - df_bt[f"pred_bt_{i}"]) / df_bt[f"true_bt_{i}"]) * 100
    norm = TwoSlopeNorm(vmin=error.min(), vcenter=0, vmax=error.max())
    plt.subplot(1, 3, 3)
    sc3 = plt.scatter(df_bt["lon_deg"], df_bt["lat_deg"], c=error, cmap="bwr", norm=norm, s=5)
    plt.colorbar(sc3)
    plt.title(f"Error % Brightness Temp - Channel {i}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(f"bt_map_channel_{i}.png")
    plt.close()
    print(f"finished ch_{i}.png")
