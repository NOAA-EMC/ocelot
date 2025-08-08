import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Load your predictions CSV
df_surface = pd.read_csv("val_csv/val_surface_obs_target_epoch75_batch0_step0.csv")
df_bt = pd.read_csv("val_csv/val_atms_target_epoch75_batch0_step0.csv")
print(df_bt.columns)

plt.figure(figsize=(16, 5))
# Get shared color scale limits
vmin = min(df_surface[f"true_ch1"].min(), df_surface[f"pred_ch1"].min())
vmax = max(df_surface[f"true_ch1"].max(), df_surface[f"pred_ch1"].max())
print("min:", vmin)
print("max:", vmax)
print("tru", df_surface[f"true_ch1"].head())
print("pred", df_surface[f"pred_ch1"].head())
print("lon:", df_surface["lon"])

# Plot target
plt.subplot(1, 3, 1)
sc1 = plt.scatter(df_surface["lon"], df_surface["lat"], c=df_surface[f"true_ch1"], cmap="viridis", s=5, vmin=vmin, vmax=vmax)
plt.colorbar(sc1)
plt.title(f"True virtualTemperature ")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Plot prediction
plt.subplot(1, 3, 2)
sc2 = plt.scatter(df_surface["lon"], df_surface["lat"], c=df_surface[f"pred_ch1"], cmap="viridis", s=5, vmin=vmin, vmax=vmax)
plt.colorbar(sc2)
plt.title(f"Predicted virtualTemperature")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

error = ((df_surface[f"true_ch1"] - df_surface[f"pred_ch1"]) / df_surface[f"true_ch1"])*100
norm = TwoSlopeNorm(vmin=error.min(), vcenter=0, vmax=error.max())
plt.subplot(1, 3, 3)
sc3 = plt.scatter(df_surface["lon"], df_surface["lat"], c=error, cmap="bwr", norm=norm, s=5)
plt.colorbar(sc3)
plt.title(f"% Error")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.savefig("virtualTemperature.png")
plt.close()
print(f"finished virtualTemperature.png")

# Plot predictions vs targets for each channel
for i in range(1, 23):  # Channels 1 to 22
    plt.figure(figsize=(16, 5))
    # Get shared color scale limits
    vmin = min(df_bt[f"true_ch{i}"].min(), df_bt[f"pred_ch{i}"].min())
    vmax = max(df_bt[f"true_ch{i}"].max(), df_bt[f"pred_ch{i}"].max())
    print("min:", vmin)
    print("max:", vmax)
    print("tru", df_bt[f"true_ch{i}"].head())
    print("pred", df_bt[f"pred_ch{i}"].head())
    print("lon:", df_bt["lon"])

    # Plot target
    plt.subplot(1, 3, 1)
    sc1 = plt.scatter(df_bt["lon"], df_bt["lat"], c=df_bt[f"true_ch{i}"], cmap="viridis", s=5, vmin=vmin, vmax=vmax)
    plt.colorbar(sc1)
    plt.title(f"True Brightness Temp - Channel {i}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Plot prediction
    plt.subplot(1, 3, 2)
    sc2 = plt.scatter(df_bt["lon"], df_bt["lat"], c=df_bt[f"pred_ch{i}"], cmap="viridis", s=5, vmin=vmin, vmax=vmax)
    plt.colorbar(sc2)
    plt.title(f"Predicted Brightness Temp - Channel {i}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    error = ((df_bt[f"true_ch{i}"] - df_bt[f"pred_ch{i}"]) / df_bt[f"true_ch{i}"])*100
    norm = TwoSlopeNorm(vmin=error.min(), vcenter=0, vmax=error.max())
    plt.subplot(1, 3, 3)
    sc3 = plt.scatter(df_bt["lon"], df_bt["lat"], c=error, cmap="bwr", norm=norm, s=5)
    plt.colorbar(sc3)
    plt.title(f" % Error Brightness Temp - Channel {i}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(f"bt_map_channel_{i}.png")
    plt.close()
    print(f"finished ch_{i}.png")
