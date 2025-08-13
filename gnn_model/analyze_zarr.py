import zarr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# --- Configuration ---
# Name of your Zarr file directory (e.g., "my_data.zarr")
ZARR_FILE_PATH = '/scratch3/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v5/global/surface_obs.zarr'
# --- End Configuration ---


def analyze_zarr_data(file_path):
    """
    Opens a Zarr file, prints descriptive statistics, and plots the spatial
    distribution of the data points on a world map.
    """
    try:
        # 1. Open the Zarr file
        store = zarr.open(file_path, mode='r')
        print(f"Successfully opened Zarr file: {file_path}")
        print("-" * 30)

        # 2. Load data into a Pandas DataFrame for analysis
        variables_to_load = [
            'latitude', 'longitude', 'height', 'airTemperature', 'airPressure',
            'seaTemperature', 'stationElevation', 'eastwardWind', 'virtualTemperature', 'northwardWind'
        ]

        data = {}
        for var in variables_to_load:
            if var in store:
                data[var] = store[var][:]
            else:
                print(f"Warning: Variable '{var}' not found in Zarr file.")

        df = pd.DataFrame(data)

        # 3. Print Descriptive Statistics
        print("## Data Statistics ##")
        print(df.describe())
        print("-" * 30)

        # 4. Create a Map of Data Locations
        print("\n## Generating Map ##")
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        ax.set_title('Spatial Distribution of Data Points')
        ax.stock_img()  # Adds a simple background map image
        ax.coastlines()  # Draws coastlines for context
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        # Create a scatter plot of all data points
        ax.scatter(df['longitude'], df['latitude'],
                   color='blue',
                   s=1.0,          # Small size for each point
                   alpha=0.3,      # Use transparency to see density
                   transform=ccrs.PlateCarree())  # Specify the data's coordinate system

        output_filename = 'data_distribution_map.png'
        plt.savefig(output_filename, dpi=200, bbox_inches='tight')

        print(f"Success! Map has been saved as: {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the ZARR_FILE_PATH is correct and the necessary libraries are installed.")


if __name__ == '__main__':
    # Set the path to the Zarr file at the top of the script
    analyze_zarr_data(ZARR_FILE_PATH)
