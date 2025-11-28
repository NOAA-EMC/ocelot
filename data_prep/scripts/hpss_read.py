# Utility to read files from a High Performance Storage System (HPSS)
import os
import subprocess
from datetime import datetime, timedelta
import argparse


def print_archive_files(year: int, output_path: str) -> None:
    # use hsi ls command to list all the files in the template path that have "gdas" in the filename
    start = datetime(year=year, month=1, day=1)
    end = datetime(year=year, month=12, day=31)
    current_date = start
    delta = timedelta(days=1)

    with open(os.path.join(output_path, f"hpss_archives_{year}.txt"), "w") as output_file:
        while current_date <= end:
            year_str = f"{current_date.year:04d}"
            month_str = f"{current_date.month:02d}"
            day_str = f"{current_date.day:02d}"
            path = os.path.join(HpssPathTemplate.format(
                year=year_str,
                month=month_str,
                day=day_str
            ), "*gdas*")
            cmd = ["hsi", "ls", path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output_file.write(result.stdout)
            current_date += delta


HpssPathTemplate = "/NCEPPROD/hpssprod/runhistory/rh{year}/{year}{month}/{year}{month}{day}"

class HpssFilenameMap:
    def __init__(self):
        self.map  = [
            (datetime(2000, 1, 1), datetime(2006, 6, 30),
                         "gpfs_hps_nco_ops_com_gfs_prod_gdas.{year}{month}{day}{hour}.tar"),
            (datetime(2019, 5, 1), datetime(2021, 4, 22),
                         "gpfs_dell1_nco_ops_com_gfs_prod_gdas.{year}{month}{day}_{hour}.tar"),
            (datetime(2019, 5, 1), datetime(2021, 4, 22),
                         "com_gfs_prod_gdas.{year}{month}{day}_{hour}.gdas.tar"),
        ]

    def get(self, for_time: datetime) -> str:
        for start, end, filename in self.map:
            if start <= for_time <= end:
                return filename
        raise ValueError(f"No filename mapping found for time {for_time}")





def make_file_list(year: int) -> list[str]:
    file_list = []
    hours = ["00", "06", "12", "18"]
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    delta = timedelta(days=1)
    current_date = start_date
    while current_date <= end_date:
        template = os.path.join(HpssPathTemplate, FileName2)

        year_str = f"{current_date.year:04d}"
        month_str = f"{current_date.month:02d}"
        day_str = f"{current_date.day:02d}"
        for hour in hours:
            file_path = template.format(
                year=year_str,
                month=month_str,
                day=day_str,
                hour=hour
            )
            file_list.append(file_path)
        current_date += delta

    return file_list

def file_list_name(year: int) -> str:
    return f'htar_list_{year}.txt'

def main(year: int, output_dir: str) -> None:

    if not os.path.exists(file_list_name(year)):
        file_list = make_file_list(year)
        with open(file_list_name(year), "w") as f:
            for file_path in file_list:
                f.write(f"{file_path}\n")

    # open the file list then for each line read the file with htar and then delete the line from the file
    with open(file_list_name(year), "r") as f:
        lines = f.readlines()

    for line in lines:
        file_path = line.strip()

        # Get the list of target files to extract
        cmd = ["htar", "-tf", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        all_files = result.stdout.splitlines()
        filtered_files = [f for f in all_files if f.endswith(('.bufr_d', '.prepbufr', '.prepbufr.acft_profiles'))]

        # Create a temporary file to hold the filtered file list
        target_files_path = f"target_files_{year}.txt"
        with open(target_files_path, "w") as temp_f:
            for f in filtered_files:
                temp_f.write(f"{f}\n")

        # Download the target files using htar
        cmd_extract = ["htar", "-xvf", file_path, "-L", target_files_path]
        subprocess.run(cmd_extract, cwd=os.path.abspath(output_dir), check=True)

        # Remove the archive we just processed from the file list
        with open(file_list_name(year), "r") as f:
            remaining_lines = f.readlines()[1:]
        with open(file_list_name(year), "w") as f:
            f.writelines(remaining_lines)

        # Remove the temporary file list
        os.remove(target_files_path)

def _make_sbatch_script(year: int, output_dir: str) -> str:
    script_content = \
f"""#!/bin/bash
#SBATCH --job-name=ocelot_hpss_read_{year}
#SBATCH --output=hpss_read_{year}.out
#SBATCH --error=hpss_read_{year}.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

module load hpss
python hpss_read.py {year} {output_dir}
"""
    script_filename = f"hpss_read_{year}.sbatch"
    with open(script_filename, "w") as f:
        f.write(script_content)
    return script_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read files from HPSS")
    parser.add_argument("year", help="Year ex: 2025", type=int)
    parser.add_argument("output_path", help="Output directory to save files")
    parser.add_argument("--batch", "-b", action="store_true", help="Run using SLURM sbatch script")
    parser.add_argument("--print_archives", action="store_true", help="Slurm batch script")
    args = parser.parse_args()

    if args.print_archives:
        print_archive_files(args.year, args.output_path)
    elif args.batch:
        sbatch_script = _make_sbatch_script(args.year, args.output_path)
        subprocess.run(["sbatch", sbatch_script], check=True)
    else:
        main(args.year, args.output_path)

