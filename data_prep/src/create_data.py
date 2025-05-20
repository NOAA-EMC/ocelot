import os
import sys
import argparse
from datetime import datetime, timedelta
import numpy as np
import bufr

sys.path.insert(0, os.path.realpath('./'))
from zarr_encoder import Encoder  # noqa: E402
import data_reader  # noqa: E402
import settings  # noqa: E402


def create_data_for_day(comm, date: datetime, type: str, suffix: str = None, append: bool = True):
    start_datetime = date
    end_datetime = date + timedelta(hours=23, minutes=59, seconds=59)

    parameters = data_reader.Parameters()
    parameters.start_time = start_datetime
    parameters.stop_time = end_datetime

    description, container = data_reader.run(comm, type, parameters)

    if container is None:
        raise ValueError("No data found")

    # Filter data based on the specified latitude and longitude ranges
    # if the settings have been defined
    if hasattr(settings, 'LAT_RANGE') and hasattr(settings, 'LON_RANGE'):
        latitudes = container.get('latitude')
        longitudes = container.get('longitude')

        mask = np.array([True] * len(latitudes))
        mask[latitudes < settings.LAT_RANGE[0]] = False
        mask[latitudes > settings.LAT_RANGE[1]] = False
        mask[longitudes < settings.LON_RANGE[0]] = False
        mask[longitudes > settings.LON_RANGE[1]] = False

        if not np.any(mask):
            return  # No data in the region

        container = container.apply_mask(mask)

    if comm.rank() == 0:
        if suffix:
            output_path = os.path.join(settings.OUTPUT_PATH, f'{type}_{suffix}.zarr')
        else:
            output_path = os.path.join(settings.OUTPUT_PATH, f'{type}.zarr')

        Encoder(description).encode(container, output_path, append=append)
        print(f"Output written to {output_path}")
        sys.stdout.flush()


def create_data(start_date: datetime,
                end_date: datetime,
                type: str,
                suffix: str = None,
                append: bool = True):
    date = start_date
    day = timedelta(days=1)

    bufr.mpi.App(sys.argv)
    comm = bufr.mpi.Comm("world")

    while date <= end_date:
        create_data_for_day(comm, date, type, suffix, append)
        date += day


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('start_date')
    parser.add_argument('end_date')
    parser.add_argument('type')
    parser.add_argument('-s', '--suffix', required=False)
    parser.add_argument('-a', '--append', required=False, default=True)

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    create_data(start_date, end_date, args.type, args.suffix, args.append)
