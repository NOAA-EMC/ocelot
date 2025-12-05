"""
Utility to retrieve files from a High Performance Storage System (HPSS).

This script handles three main operations for downloading GDAS data from NOAA RDHPCS HPSS:
1. Generate lists of files to retrieve (accounting for filename changes across time periods)
2. Stage files on HPSS using 'hsi' command
3. Download staged files using 'htar' command

Usage:
    python hpss_read.py <year> <output_dir> [options]

Examples:
    # Generate file lists only
    python hpss_read.py 2020 /path/to/output --generate

    # Stage files for retrieval
    python hpss_read.py 2020 /path/to/output --stage

    # Download staged files
    python hpss_read.py 2020 /path/to/output --download

    # Do everything (generate, stage, download)
    python hpss_read.py 2020 /path/to/output --all
"""
import os
import subprocess
import sys
from datetime import datetime, timedelta
import argparse

HpssPathTemplate = "/NCEPPROD/hpssprod/runhistory/rh{year}/{year}{month}/{year}{month}{day}"

class HpssFilePath:
    """
    Maps dates to HPSS archive filename templates.
    Handles filename changes across different time periods.
    """
    def __init__(self):
        # Time periods with different filename patterns
        self.map = [
            (datetime(2015, 1, 1), datetime(2016, 5, 9),
             "com_gfs_prod_gdas.{year}{month}{day}{hour}.tar"),
            (datetime(2016, 5, 10), datetime(2017, 7, 19),
             "com2_gfs_prod_gdas.{year}{month}{day}{hour}.tar"),
            (datetime(2017, 7, 20), datetime(2019, 6, 11),
             "gpfs_hps_nco_ops_com_gfs_prod_gdas.{year}{month}{day}{hour}.tar"),
            (datetime(2019, 6, 12), datetime(2020, 2, 25),
             "gpfs_dell1_nco_ops_com_gfs_prod_gdas.{year}{month}{day}_{hour}.gdas.tar"),
            (datetime(2020, 2, 26), datetime.now(),
             "com_gfs_prod_gdas.{year}{month}{day}_{hour}.gdas.tar"),
        ]

    def get(self, for_date: datetime) -> str:
        """Get the HPSS file path for a given date."""
        template = None
        for start, end, filename in self.map:
            if start <= for_date <= end + timedelta(hours=18):
                template = os.path.join(HpssPathTemplate, filename)

        if not template:
            raise ValueError(f"No template found for date {for_date}")

        return template.format(year=f"{for_date.year:04d}",
                               month=f"{for_date.month:02d}",
                               day=f"{for_date.day:02d}",
                               hour=f"{for_date.hour:02d}")


hpss_file_path = HpssFilePath()


def file_list_name(year: int) -> str:
    """Get the filename for the htar file list."""
    return f'htar_{year}.txt'


def stage_file_name(year: int) -> str:
    """Get the filename for the staging command list."""
    return f'stage_{year}.txt'


def make_list_file(year: int) -> str:
    """
    Generate a list of HPSS archive files to retrieve for the given year.
    Creates a file named 'htar_<year>.txt' with one file path per line.
    
    Returns:
        The name of the created file list.
    """
    file_list = []
    hours = ["00", "06", "12", "18"]
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23, 59, 59)
    delta = timedelta(days=1)
    current_date = start_date
    
    print(f"Generating file list for year {year}...")
    while current_date <= end_date:
        for hour in hours:
            current_date = current_date.replace(hour=int(hour))
            try:
                file_path = hpss_file_path.get(current_date)
                file_list.append(file_path)
            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)
        current_date += delta

    filename = file_list_name(year)
    with open(filename, "w") as output_file:
        for file_path in file_list:
            output_file.write(f"{file_path}\n")

    print(f"Created file list: {filename} ({len(file_list)} files)")
    return filename


def make_stage_file(year: int) -> str:
    """
    Generate a staging command file for HPSS.
    Creates a file named 'stage_<year>.txt' with 'stage <path>' commands.
    
    This file can be used with: hsi "in < stage_<year>.txt"
    
    Returns:
        The name of the created staging file.
    """
    list_file = file_list_name(year)
    if not os.path.exists(list_file):
        print(f"Error: File list {list_file} not found. Run with --generate first.", file=sys.stderr)
        sys.exit(1)

    with open(list_file, "r") as f:
        file_list = [line.strip() for line in f.readlines()]

    filename = stage_file_name(year)
    with open(filename, "w") as f:
        for file_path in file_list:
            f.write(f"stage {file_path}\n")

    print(f"Created staging file: {filename} ({len(file_list)} files)")
    print(f"To stage files, run: hsi 'in < {filename}'")
    return filename


def stage_files(year: int) -> None:
    """
    Stage files on HPSS by running hsi command.
    This prepares files for faster retrieval.
    """
    stage_file = stage_file_name(year)
    if not os.path.exists(stage_file):
        print(f"Error: Staging file {stage_file} not found. Run with --generate first.", file=sys.stderr)
        sys.exit(1)

    print(f"Staging files from {stage_file}...")
    print("This may take a while. Files will be queued for staging on HPSS.")
    
    # Run hsi with the staging commands
    cmd = ["hsi", f"in < {stage_file}"]
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("Staging complete!")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error staging files: {e}", file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(1)



def download_files(year: int, output_dir: str) -> None:
    """
    Download files from HPSS using htar.
    Extracts only specific file types (.bufr_d, .prepbufr, .prepbufr.acft_profiles).
    
    Args:
        year: Year of data to download
        output_dir: Directory where files will be extracted
    """
    list_file = file_list_name(year)
    if not os.path.exists(list_file):
        print(f"Error: File list {list_file} not found. Run with --generate first.", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    with open(list_file, "r") as f:
        archive_files = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Processing {len(archive_files)} archive files...")
    
    for idx, archive_path in enumerate(archive_files, 1):
        print(f"\n[{idx}/{len(archive_files)}] Processing: {archive_path}")
        
        try:
            # Get the list of files in the archive
            print("  Listing archive contents...")
            cmd = ["htar", "-tf", archive_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            all_files = result.stdout.splitlines()
            
            # Filter for desired file types
            filtered_files = [f for f in all_files if f.endswith(('.bufr_d', '.prepbufr', '.prepbufr.acft_profiles'))]
            
            if not filtered_files:
                print(f"  No matching files found in archive, skipping.")
                continue
            
            print(f"  Found {len(filtered_files)} matching files to extract")
            
            # Create a temporary file list for extraction
            target_files_path = f"/tmp/target_files_{year}_{idx}.txt"
            with open(target_files_path, "w") as temp_f:
                for f in filtered_files:
                    temp_f.write(f"{f}\n")
            
            # Extract the files
            print("  Extracting files...")
            cmd = ["htar", "-xvf", archive_path, "-L", target_files_path]
            subprocess.run(cmd, cwd=os.path.abspath(output_dir), check=True, capture_output=True)
            
            # Clean up temporary file list
            os.remove(target_files_path)
            print("  Extraction complete")
            
        except subprocess.CalledProcessError as e:
            print(f"  Error processing archive: {e}", file=sys.stderr)
            if e.stderr:
                print(f"  {e.stderr}", file=sys.stderr)
            print("  Skipping to next archive...")
            continue
        except Exception as e:
            print(f"  Unexpected error: {e}", file=sys.stderr)
            print("  Skipping to next archive...")
            continue

    print(f"\nDownload complete! Files extracted to: {output_dir}")



def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Retrieve GDAS data from NOAA RDHPCS HPSS system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate file lists for 2020
  %(prog)s 2020 /data/output --generate

  # Stage files for retrieval (after generating)
  %(prog)s 2020 /data/output --stage

  # Download files (after staging)
  %(prog)s 2020 /data/output --download

  # Do everything in one command
  %(prog)s 2020 /data/output --all

Workflow:
  1. Generate: Creates htar_<year>.txt and stage_<year>.txt files
  2. Stage: Runs 'hsi' to stage files on HPSS for faster retrieval
  3. Download: Uses 'htar' to extract files from archives
        """
    )
    
    parser.add_argument("year", type=int, help="Year of data to retrieve (e.g., 2020)")
    parser.add_argument("output_dir", help="Directory where files will be extracted")
    
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--generate", "-g", action="store_true",
                              help="Generate file lists (htar_<year>.txt and stage_<year>.txt)")
    action_group.add_argument("--stage", "-s", action="store_true",
                              help="Stage files on HPSS (requires --generate first)")
    action_group.add_argument("--download", "-d", action="store_true",
                              help="Download files from HPSS (requires --generate and --stage first)")
    action_group.add_argument("--all", "-a", action="store_true",
                              help="Do everything: generate, stage, and download")
    
    args = parser.parse_args()
    
    # Validate year
    current_year = datetime.now().year
    if args.year < 2015 or args.year > current_year:
        print(f"Error: Year must be between 2015 and {current_year}", file=sys.stderr)
        sys.exit(1)
    
    # Execute requested actions
    try:
        if args.all:
            print("=" * 60)
            print(f"HPSS Retrieval - Year {args.year}")
            print("=" * 60)
            print("\nStep 1: Generating file lists...")
            make_list_file(args.year)
            make_stage_file(args.year)
            
            print("\nStep 2: Staging files on HPSS...")
            stage_files(args.year)
            
            print("\nStep 3: Downloading files...")
            download_files(args.year, args.output_dir)
            
            print("\n" + "=" * 60)
            print("All operations complete!")
            print("=" * 60)
            
        elif args.generate:
            make_list_file(args.year)
            make_stage_file(args.year)
            
        elif args.stage:
            stage_files(args.year)
            
        elif args.download:
            download_files(args.year, args.output_dir)
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


