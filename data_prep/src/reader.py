import os
import sys
import argparse
from datetime import datetime, timedelta
import numpy as np
import bufr

sys.path.insert(0, os.path.realpath('/'))
from zarr_encoder import Encoder as ZarrEncoder  # noqa: E402
from parquet_encoder import Encoder as ParquetEncoder  # noqa: E402
import runner  # noqa: E402
import settings  # noqa: E402


def create_data(start_date: datetime,
                end_date: datetime,
                data_type: str,
                output_type: str,
                suffix: str = None,
                append: bool = True) -> None:
    """
    Create data files from BUFR data for each day in the specified date range.

    Parameters
    ----------
    start_date : datetime
        Start date (inclusive).
    end_date : datetime
        End date (inclusive).
    data_type : str
        Data type to process (must be defined in config).
    output_type : str
        Output file type ('zarr' or 'parquet').
    suffix : str, optional
        Suffix to append to the output file name.
    append : bool, optional
    """

    bufr.mpi.App(sys.argv)
    comm = bufr.mpi.Comm("world")

    extension = 'zarr' if output_type == 'zarr' else 'pqt'

    if suffix:
        file_name = f"{data_type}_{suffix}.{extension}"
    else:
        file_name = f"{data_type}.{extension}"

    output_path = os.path.join(settings.OUTPUT_PATH, file_name)

    if output_type == 'zarr':
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
        _append_data_for_day(comm, date, data_type, output_type, output_path)
        date += day


def create_weekly_data(start_date: datetime,
                       end_date: datetime,
                       data_type: str,
                       output_type: str = 'parquet',
                       suffix: str = None,
                       append: bool = True) -> None:

    """
    Create data files from BUFR data for each week in the specified date range.

    Parameters
    ----------
    start_date : datetime
        Start date (inclusive).
    end_date : datetime
        End date (inclusive).
    data_type : str
        Data type to process (must be defined in config).
    output_type : str
        Output file type ('zarr' or 'parquet').
    suffix : str, optional
        Suffix to append to the output file name.
    append : bool, optional
    """

    bufr.mpi.App(sys.argv)
    comm = bufr.mpi.Comm("world")

    # Determine all week boundaries (Monday - Sunday) that intersect the range
    week_start = start_date - timedelta(days=start_date.weekday())
    week_ranges = []
    while week_start <= end_date:
        week_end = week_start + timedelta(days=6)
        week_ranges.append((week_start, week_end))
        week_start = week_end + timedelta(days=1)

    extension = 'zarr' if output_type == 'zarr' else 'pqt'

    # Generate output paths for each week
    output_paths = {}
    for wstart, wend in week_ranges:
        if suffix:
            file_name = f"{data_type}_{suffix}_{wstart:%Y%m%d}_{wend:%Y%m%d}.{extension}"
        else:
            file_name = f"{data_type}_{wstart:%Y%m%d}_{wend:%Y%m%d}.{extension}"
        output_paths[(wstart, wend)] = os.path.join(settings.OUTPUT_PATH, file_name)

    if output_type == 'zarr':
        if comm.rank() == 0:
            # Ensure all output directories exist before processing
            for path in output_paths.values():
                if not append and os.path.exists(path):
                    import shutil
                    shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)

        comm.barrier()

    # Process each day and append to the appropriate weekly file
    day = timedelta(days=1)
    date = start_date
    while date <= end_date:
        week_start = date - timedelta(days=date.weekday())
        week_end = week_start + timedelta(days=6)
        out_path = output_paths[(week_start, week_end)]

        _append_data_for_day(comm, date, data_type, output_type, out_path)
        date += day


def create_monthly_data(start_date: datetime,
                        end_date: datetime,
                        data_type: str,
                        output_type: str = 'zarr',
                        suffix: str = None,
                        append: bool = True) -> None:
    """Create zarr files from BUFR data in month long chunks."""

    bufr.mpi.App(sys.argv)
    comm = bufr.mpi.Comm("world")

    # Determine all month boundaries that intersect the range
    month_ranges = []
    current = start_date.replace(day=1)
    while current <= end_date:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        month_end = next_month - timedelta(days=1)
        month_ranges.append((current, month_end))
        current = next_month

    extension = 'zarr' if output_type == 'zarr' else 'pqt'

    # Generate output paths for each month
    output_paths = {}
    for mstart, mend in month_ranges:
        if suffix:
            file_name = f"{data_type}_{suffix}_{mstart:%Y%m}.{extension}"
        else:
            file_name = f"{data_type}_{mstart:%Y%m}.{extension}"
        output_paths[(mstart, mend)] = os.path.join(settings.OUTPUT_PATH, file_name)

    if output_type == 'zarr':
        if comm.rank() == 0:
            # Ensure all output directories exist before processing
            for path in output_paths.values():
                if not append and os.path.exists(path):
                    import shutil
                    shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)
        comm.barrier()

    # Process each day and append to the appropriate monthly file
    day = timedelta(days=1)
    date = start_date
    while date <= end_date:
        month_start = date.replace(day=1)
        next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        month_end = next_month - timedelta(days=1)
        out_path = output_paths[(month_start, month_end)]

        _append_data_for_day(comm, date, data_type, output_type, out_path)
        date += day


def create_yearly_data(start_date: datetime,
                       end_date: datetime,
                       data_type: str,
                       output_type: str = 'zarr',
                       suffix: str = None,
                       append: bool = True) -> None:
    """
    Create zarr files from BUFR data in year long chunks.
    """

    bufr.mpi.App(sys.argv)
    comm = bufr.mpi.Comm("world")

    # Determine all year boundaries that intersect the range
    year_ranges = []
    current = start_date.replace(month=1, day=1)
    while current <= end_date:
        year_end = current.replace(month=12, day=31)
        year_ranges.append((current, year_end))
        current = current.replace(year=current.year + 1, month=1, day=1)

    extension = 'zarr' if output_type == 'zarr' else 'pqt'

    # Generate output paths for each year
    output_paths = {}
    for ystart, yend in year_ranges:
        if suffix:
            file_name = f"{data_type}_{suffix}_{ystart:%Y}.{extension}"
        else:
            file_name = f"{data_type}_{ystart:%Y}.{extension}"
        output_paths[(ystart, yend)] = os.path.join(settings.OUTPUT_PATH, file_name)

    if output_type == 'zarr':
        if comm.rank() == 0:
            # Ensure all output directories exist before processing
            for path in output_paths.values():
                if not append and os.path.exists(path):
                    import shutil
                    shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)
        comm.barrier()

    # Process each day and append to the appropriate yearly file
    day = timedelta(days=1)
    date = start_date
    while date <= end_date:
        year_start = date.replace(month=1, day=1)
        year_end = date.replace(month=12, day=31)
        out_path = output_paths[(year_start, year_end)]

        _append_data_for_day(comm, date, data_type, output_type, out_path)
        date += day


def _append_data_for_day(comm,
                         date: datetime,
                         data_type: str,
                         output_type: str,
                         output_path: str) -> None:

    start_datetime = date
    end_datetime = date + timedelta(hours=23, minutes=59, seconds=59)

    parameters = runner.Parameters()
    parameters.start_time = start_datetime
    parameters.stop_time = end_datetime

    description, container = runner.run(comm, data_type, parameters)

    if comm.rank() == 0:
        if container is None or container.size() == 0:
            return  # No data for this day

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

        if output_type == 'zarr':
            ZarrEncoder(description).encode(container, f'{output_path}', append=True)
        elif output_type == 'parquet':
            ParquetEncoder(description).encode(container, f'{output_path}', append=True)
        else:
            raise ValueError(f"Unsupported output type: {output_type}")

        print(f"Output written to {output_path}")
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('start_date')
    parser.add_argument('end_date')
    parser.add_argument('type')
    parser.add_argument('output_type', choices=['zarr', 'parquet'], help='Output file type')
    parser.add_argument('-s', '--suffix', required=False, help='Suffix for the output file(s)')
    parser.add_argument('-a', '--append', action='store_true', help='Append to existing data')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    if args.output_type == 'zarr':
        create_yearly_data(start_date, end_date, args.type, args.output_type, args.suffix, args.append)
    elif args.output_type == 'parquet':
        create_weekly_data(start_date, end_date, args.type, args.output_type, args.suffix, args.append)
