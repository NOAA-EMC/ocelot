# Utility to read files from a High Performance Storage System (HPSS)
import os
import subprocess
from datetime import datetime, timedelta
import argparse

HpssPathTemplate = "/NCEPPROD/hpssprod/runhistory/rh{year}/{year}{month}/{year}{month}{day}"\
                   "/com_gfs_prod_gdas.{year}{month}{day}_{hour}.gdas.tar"

def make_file_list(template: str, year: int) -> list[str]:
    file_list = []
    hours = ["00", "06", "12", "18"]
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    delta = timedelta(days=1)
    current_date = start_date
    while current_date <= end_date:
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
        file_list = make_file_list(HpssPathTemplate, year)
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
    parser.add_argument("year", help="Year ex: 2025")
    parser.add_argument("output_dir", help="Output directory to save files")
    parser.add_argument("batch", "-b", action="store_true", help="Run using SLURM sbatch script")
    args = parser.parse_args()

    if args.batch:
        sbatch_script = _make_sbatch_script(args.year, args.output_dir)
        subprocess.run(["sbatch", sbatch_script], check=True)
    else:
        main(args.year, args.output_dir)

