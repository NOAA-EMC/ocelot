import os
import sys
import argparse
from datetime import datetime, timedelta
import numpy as np
import bufr

sys.path.insert(0, os.path.realpath('/'))
from zarr_encoder import Encoder  # noqa: E402
import data_reader  # noqa: E402
import settings  # noqa: E402


def create_data(start_date: datetime,
                end_date: datetime,
                data_type: str,
                suffix: str = None,
                append: bool = True) -> None:
    """Create zarr files from BUFR data in week long chunks."""

    bufr.mpi.App(sys.argv)
    comm = bufr.mpi.Comm("world")

    if suffix:
        file_name = f"{data_type}_{suffix}.zarr"
    else:
        file_name = f"{data_type}.zarr"

    output_path = os.path.join(settings.OUTPUT_PATH, file_name)

    if comm.rank() == 0:
        # Ensure all output directories exist before processing
        if not append and os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

    comm.barrier()

    date = start_date
    day = timedelta(days=1)

    while date <= end_date:
        _create_data_for_day(comm, date, data_type, output_path)
        date += day


def _create_data_for_day(comm,
                        date: datetime,
                        data_type: str,
                        output_path: str,
                        append: bool = True) -> None:
    start_datetime = date
    end_datetime = date + timedelta(hours=23, minutes=59, seconds=59)

    parameters = data_reader.Parameters()
    parameters.start_time = start_datetime
    parameters.stop_time = end_datetime

    description, container = data_reader.run(comm, data_type, parameters)

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

        container.apply_mask(mask)

    if comm.rank() == 0:
        Encoder(description).encode(container, output_path, append=append)
        print(f"Output written to {output_path}")
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('start_date')
    parser.add_argument('end_date')
    parser.add_argument('type')
    parser.add_argument('-s', '--suffix', required=False, help='Suffix for the output file(s)')
    parser.add_argument('-a', '--append', action='store_true', help='Append to existing data')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    create_data(start_date, end_date, args.type, args.suffix, args.append)
