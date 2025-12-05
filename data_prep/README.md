
# Data Preparation

The ocelot/data_prep directory contains code that is used to ingest raw data and turn it into **ocelot** training data.
The primary function for using it is `scripts/gen_data.py`. It can be used to easily generate any amount of data for 
the data type of interest. For example:

```bash
python gen_data.py -b 2024-04-01 2024-07-31 atms
```

This will generate data for the ATMS instrument for the time period of April 1, 2024 to July 31, 2024 using SLURM
sbatch. The script automatically breaks the job into smaller chunks that can be executed within the time limits of the
cluster. The script also supports parallel execution (does not chunk the job), and serial mode. Please see the help:

```
usage: gen_data.py [-h] [-s SUFFIX] [-p] [-b] [-a] [--slurm_account SLURM_ACCOUNT] start_date end_date {all,atms,surface_pressure,radiosonde}

positional arguments:
  start_date            Start date in YYYY-MM-DD format
  end_date              End date in YYYY-MM-DD format
  {all,atms,surface_pressure,radiosonde}
                        Data type to generate. "all" generates all data types.

options:
  -h, --help            show this help message and exit
  -s SUFFIX, --suffix SUFFIX
                        Suffix for the output file(s)
  -p, --parallel        Run in parallel (using either srun or mpirun)
  -b, --batch           Run in batch mode (using sbatch). 
                        Chunks the data into multiple tasks if needed.
  -a, --append          Append to existing data

  --slurm_account SLURM_ACCOUNT
                        SLURM account name for batch jobs
```

The generated Zarr files now contain at most one week of data. Files are named
`<type>_<suffix>_<YYYYMMDD>_<YYYYMMDD>.zarr` (the suffix portion is omitted if
not provided) where the dates represent the Monday-Sunday range for that week.
Subsequent runs automatically append to these files when processing additional
days within the same week.

## Configuration /  Mapping files

In order to run the script in an environment, you first need to "install" it by defining the configuration that it needs
to run. There are several configuration files you will need:

- `configs/local_settings.py`: *Note - Copy `configs/template_settings.py` to `configs/local_settings.py` to get 
started.* This configuration file defines the characteristics of the platform you are running on. For example, it 
defines the location of the source data (dump directory), the location of the type configuration yaml, the output 
directory and so on.
- `configs/<type configs>.yaml`: *Example - `configs/hera.yaml`* This file tells the system how to map raw input files 
to mapping files, with additional info for how to process handle the data. Please see `configs/hera.yaml` for an 
example.
- `mapping`: This directory contains the bufr-query mapping files that are used in the file conversion process.

## Running on HERA (using existing installation)

Since the data_prep code is already installed on HERA, you can run the script without needing to install it. In order to
use it you can do the following:

1) Log into HERA
2) `cd /scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/`
3) `source ./env.sh` To set up the environment.
4) `cd src/ocelot/data_prep/scripts`
5) `python gen_data.py -b 2024-04-01 2024-04-07 "atms" -s "my_suffix"` Please define a special suffix if you are playing
around as you will replace the existing data. Please note that the output directory for the data is defined by the
`configs/local_settings.py` file.

## HPSS Data Retrieval

The `scripts/hpss_read.py` script is used to download GDAS observation data from the NOAA RDHPCS HPSS archive system.
The script handles three main operations:

1. **Generate** file lists (accounting for filename changes across different time periods)
2. **Stage** files on HPSS for faster retrieval
3. **Download** files using htar

### Usage

```bash
# Generate file lists for a specific year
python hpss_read.py <year> <output_dir> --generate

# Stage files on HPSS (after generating)
python hpss_read.py <year> <output_dir> --stage

# Download files (after staging)
python hpss_read.py <year> <output_dir> --download

# Do everything in one command
python hpss_read.py <year> <output_dir> --all
```

### Example

```bash
# Complete workflow for 2020 data
cd /scratch1/NCEPDEV/da/$USER/hpss_data
python /path/to/ocelot/data_prep/scripts/hpss_read.py 2020 ./output_2020 --generate
python /path/to/ocelot/data_prep/scripts/hpss_read.py 2020 ./output_2020 --stage
python /path/to/ocelot/data_prep/scripts/hpss_read.py 2020 ./output_2020 --download
```

### Generated Files

- `htar_<year>.txt`: List of archive files to download (used by htar)
- `stage_<year>.txt`: Staging commands for HPSS (used by hsi)

The download process automatically extracts only the relevant observation files (.bufr_d, .prepbufr, .prepbufr.acft_profiles)
from the archives.

### Notes

- The script automatically handles filename pattern changes across different time periods (2015-present)
- Staging files on HPSS before downloading significantly speeds up retrieval
- The download process can be interrupted and resumed (it processes archives sequentially)
