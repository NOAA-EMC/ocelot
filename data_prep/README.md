
# Data Preparation

The data_prep directory contains code that is used to ingest raw data and turn it into **ocelot** training data. The
primary function for doing this is `scripts/gen_data.py`. It can be used to easily generate any amount of data for the
data type of interest. For example:

```bash
python gen_data.py -b 2024-04-01 2024-07-31 atms
```

This will generate data for the ATMS instrument for the time period of April 1, 2024 to July 31, 2024 using SLURM
sbatch. The script automatically breaks the job into smaller chunks that can be executed within the time limits of the
cluster. The script also supports parallel execution (does not chunk the job), and serial mode. Please see the help:

```
usage: gen_data.py [-h] [-s SUFFIX] [-p] [-b] [-a] start_date end_date {all,atms,surface_pressure,radiosonde}

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
```

## Configuration /  Mapping files

In order to run the script in an environment, you need to "install" it by defining the configuration it needs to run.
There are several configuration files you will need:

- `configs/local_settings.py`: This is a configuration file that tells this "installation" defines the characteristics
of the platform you are running on. For example it defines the location of the source data (dump directory), the
location of the type configuration yaml, the output directory and so on. Please see `configs/template_settings.py` for
more detailed information.
- `configs/<type configs>.yaml`: (example `configs/hera.yaml`). This file tells the system how to map raw input files to
mapping files, with additional info for how to process handle the data. Please see `configs/hera.yaml` for a complete 
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
