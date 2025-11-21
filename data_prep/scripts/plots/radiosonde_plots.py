import argparse
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.markers import MarkerStyle
import zarr

from emcpy.plots import CreatePlot, CreateFigure
from emcpy.plots.map_tools import Domain, MapProjection
from emcpy.plots.map_plots import MapScatter


def plot(domain: str = "conus"):
    parser = argparse.ArgumentParser(description="Plot Raw Radiosonde Data")
    parser.add_argument("zarr_path", help="Path to Zarr dataset")
    args = parser.parse_args()
    z = zarr.open(args.zarr_path)

    time = z['time'][:]
    lats = z['latitude'][:]
    lons = z['longitude'][:]
    pressure = z['airPressure'][:]
    flightId = z['flightId'][:]

    scatter = MapScatter(lats, lons, flightId)
    scatter.markersize = .0125
    scatter.marker = "."
    scatter.cmap = 'prism'

    # Create plot object and add features
    plot1 = CreatePlot()
    plot1.figsize = (18, 14)
    plot1.plot_layers = [scatter]
    plot1.projection = 'plcarr'
    plot1.domain = domain
    plot1.add_map_features(['coastline'])
    plot1.add_xlabel(xlabel='longitude')
    plot1.add_ylabel(ylabel='latitude')
    plot1.add_title(label='Radiosonde', loc='center', fontsize=20)

    fig = CreateFigure(figsize=(12, 10))
    fig.plot_list = [plot1]
    fig.create_figure()

    plt.savefig(f'radiosonde_{domain}.png', dpi=300)


def make_gif(domain: str = 'conus'):
    parser = argparse.ArgumentParser(description="Plot Raw Radiosonde Data")
    parser.add_argument("zarr_path", help="Path to Zarr dataset")
    args = parser.parse_args()
    z = zarr.open(args.zarr_path)

    time = z["time"][:]
    time = np.array([np.datetime64(int(t), "s") for t in time])

    lats = z["latitude"][:]
    lons = z["longitude"][:]
    pressure = z["airPressure"][:]

    first_day = time[0].astype("M8[D]").astype(datetime)

    def data_for_day(day: datetime):
        """Extract latitude, longitude and pressure for a given day."""
        day_start = np.datetime64(day, "D")
        day_end = day_start + np.timedelta64(1, "D")
        mask = (time >= day_start) & (time < day_end)
        return lats[mask], lons[mask], pressure[mask]

    # Create base plot without any scatter points
    base_plot = CreatePlot()
    base_plot.figsize = (18, 14)
    base_plot.projection = "plcarr"
    base_plot.domain = domain
    base_plot.add_map_features(["coastline"])
    base_plot.add_xlabel(xlabel="longitude")
    base_plot.add_ylabel(ylabel="latitude")
    base_plot.add_title(label="Radiosonde", loc="center", fontsize=20)

    fig = CreateFigure(figsize=(12, 10))
    fig.plot_list = [base_plot]
    fig.create_figure()
    ax = fig.fig.axes[0]

    # Initial scatter plotted using matplotlib so that it can be updated
    day_lats, day_lons, day_pressures = data_for_day(first_day)
    scat = ax.scatter(day_lons, day_lats, c=day_pressures, s=5, marker=".")

    def update(frame):
        day = first_day + timedelta(days=frame)
        day_lats, day_lons, day_pressures = data_for_day(day)

        # Update data for the existing scatter artist
        scat.set_offsets(np.column_stack((day_lons, day_lats)))
        scat.set_array(day_pressures)
        plt.suptitle(
            f"Radiosonde Data for {day.strftime('%Y-%m-%d')}", fontsize=16
        )
        return scat,

    # Get the number of days in the dataset
    num_days = len(np.unique(time.astype("M8[D]")))

    # Create animation and save as GIF
    ani = animation.FuncAnimation(fig.fig, func=update, frames=num_days, interval=200)
    ani.save(f"radiosonde_{domain}.gif", writer="imagemagick", fps=5, dpi=300)


if __name__ == '__main__':
    # plot(domain='conus')
    # plot(domain='global')
    make_gif(domain='conus')
    # make_gif(domain='global')
    # make_gif(domain='europe')
    # make_gif(domain='northeast')
