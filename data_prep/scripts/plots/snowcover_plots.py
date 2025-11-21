"""
Plotting 2D scatter data on map plot
------------------------------------

Sometimes, users only want to look at the locations
of their data on a map and do not care about the
actual values. Below is an example of how to just plot
lat and lon values on map.

"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from emcpy.plots import CreatePlot, CreateFigure
from emcpy.plots.map_tools import Domain, MapProjection
from emcpy.plots.map_plots import MapScatter

import zarr


def main():
    parser = argparse.ArgumentParser(description="Summarize variables in a Zarr file")
    parser.add_argument("zarr_path", help="Path to Zarr dataset")
    args = parser.parse_args()
    z = zarr.open(args.zarr_path)
    temp = z['snowDepth'][:]
    lats = z['latitude'][:]
    lons = z['longitude'][:]
    # temp = z['airTemperature'][1000000:1010000]

    lats = lats[temp < 99999]
    lons = lons[temp < 99999]
    temp = temp[temp < 99999]

    print(temp.max(), temp.min(), temp.mean())

    scatter = MapScatter(lats, lons, data=temp)
    scatter.markersize = .25

    # Create plot object and add features
    plot1 = CreatePlot()
    plot1.plot_layers = [scatter]
    plot1.projection = 'plcarr'
    plot1.domain = 'global'
    plot1.add_map_features(['coastline'])
    plot1.add_xlabel(xlabel='longitude')
    plot1.add_ylabel(ylabel='latitude')
    plot1.add_title(label='EMCPy Map', loc='center', fontsize=20)

    fig = CreateFigure(figsize=(12, 10))
    fig.plot_list = [plot1]
    fig.create_figure()

    plt.savefig('test.png')


if __name__ == '__main__':
    main()
