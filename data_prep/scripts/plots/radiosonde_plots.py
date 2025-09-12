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

def plot_global():
    parser = argparse.ArgumentParser(description="Plot Raw Radiosonde Data")
    parser.add_argument("zarr_path", help="Path to Zarr dataset")
    args = parser.parse_args()
    z = zarr.open(args.zarr_path)

    time = z['time'][:]
    lats = z['latitude'][:]
    lons = z['longitude'][:]
    pressure = z['airPressure'][:]

    scatter = MapScatter(lats, lons, time)
    scatter.markersize = .0125
    scatter.marker = "*"

    # Create plot object and add features
    plot1 = CreatePlot()
    plot1.figsize = (18, 14)
    plot1.plot_layers = [scatter]
    plot1.projection = 'plcarr'
    plot1.domain = 'global'
    plot1.add_map_features(['coastline'])
    plot1.add_xlabel(xlabel='longitude')
    plot1.add_ylabel(ylabel='latitude')
    plot1.add_title(label='Radiosonde', loc='center', fontsize=20)

    fig = CreateFigure(figsize=(12, 10))
    fig.plot_list = [plot1]
    fig.create_figure()
    # plt.show()

    plt.savefig('radiosonde_global.png', dpi=300)

def plot_conus():
    parser = argparse.ArgumentParser(description="Plot Raw Radiosonde Data")
    parser.add_argument("zarr_path", help="Path to Zarr dataset")
    args = parser.parse_args()
    z = zarr.open(args.zarr_path)

    time = z['time'][:]
    lats = z['latitude'][:]
    lons = z['longitude'][:]
    pressure = z['airPressure'][:]

    scatter = MapScatter(lats, lons, time)
    scatter.markersize = .0125
    scatter.marker = "*"

    # Create plot object and add features
    plot1 = CreatePlot()
    plot1.figsize = (18, 14)
    plot1.plot_layers = [scatter]
    plot1.projection = 'plcarr'
    plot1.domain = 'conus'
    plot1.add_map_features(['coastline'])
    plot1.add_xlabel(xlabel='longitude')
    plot1.add_ylabel(ylabel='latitude')
    plot1.add_title(label='Radiosonde', loc='center', fontsize=20)

    fig = CreateFigure(figsize=(12, 10))
    fig.plot_list = [plot1]
    fig.create_figure()
    # plt.show()

    plt.savefig('radiosonde_conus.png', dpi=300)

def make_gif():
    parser = argparse.ArgumentParser(description="Plot Raw Radiosonde Data")
    parser.add_argument("zarr_path", help="Path to Zarr dataset")
    args = parser.parse_args()
    z = zarr.open(args.zarr_path)

    time = z['time'][:]
    time = np.array([np.datetime64(int(t), 's') for t in time])

    lats = z['latitude'][:]
    lons = z['longitude'][:]
    pressure = z['airPressure'][:]

    first_day = time[0].astype('M8[D]').astype(datetime)

    def init():
        day_times, day_lats, day_lons, day_pressures = data_for_day(time[0].astype('M8[D]').astype(datetime))
        scatter = MapScatter(day_lats, day_lons, day_pressures)
        scatter.markersize = .0125
        scatter.marker = "*"

        # Create plot object and add features
        plot1 = CreatePlot()
        plot1.figsize = (18, 14)
        plot1.plot_layers = [scatter]
        plot1.projection = 'plcarr'
        plot1.domain = 'conus'
        plot1.add_map_features(['coastline'])
        plot1.add_xlabel(xlabel='longitude')
        plot1.add_ylabel(ylabel='latitude')
        plot1.add_title(label='Radiosonde', loc='center', fontsize=20)

        fig = CreateFigure(figsize=(12, 10))
        fig.plot_list = [plot1]
        fig.create_figure()
        # plt.show()

        print(dir(fig))

        return fig, plot1, scatter

    def data_for_day(day:datetime):
        # Extract data for the specified day

        day_start = np.datetime64(day, 'D')
        day_end = day_start + np.timedelta64(1, 'D')
        mask = (time >= day_start) & (time < day_end)
        return time[mask], lats[mask], lons[mask], pressure[mask]


    fig, plot1, scatter = init()
    print(dir(plot1))

    def update(frame):
        day = first_day + timedelta(days=frame)
        day_times, day_lats, day_lons, day_pressures = data_for_day(day)

        print ('frame:', day)
        # print ('lats:', day_lats)

        scatter = MapScatter(day_lats, day_lons, day_pressures)
        scatter.markersize = .0125
        scatter.marker = "*"

        plot1.plot_layers = [scatter]

        plt.suptitle(f'Radiosonde Data for {day.strftime("%Y-%m-%d")}', fontsize=16)

        return scatter,

    # Get the number of days in time
    num_days = len(np.unique(time.astype('M8[D]')))

    # make gif animation
    ani = animation.FuncAnimation(fig.fig, func=update, frames=num_days, interval=200)
    ani.save('radiosonde.gif', writer='imagemagick', fps=5, dpi=300)

    # plt.savefig('radiosonde.png', dpi=300)

if __name__ == '__main__':
    # plot_global()
    # plot_conus()
    make_gif()
