import sys
import os
import argparse
from datetime import datetime, timedelta

base_path = os.path.split(os.path.realpath(__file__))[0]

sys.path.append(os.path.realpath(os.path.join(base_path, '..', 'src')))


def get_reader_path(data_type: str) -> str:
    """Return path to the unified reader script."""
    return os.path.realpath(os.path.join(base_path, '../src/reader.py'))


def _is_slurm_available() -> bool:
    """
    Check if SLURM is available.
    """
    try:
        import subprocess
        result = subprocess.run(['srun', '-h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except Exception:
        return False


def _make_sbatch_cmd(reader_path: str,
                     idx: int,
                     start: datetime,
                     end: datetime,
                     ntasks: int,
                     data_type: str,
                     output_type: str,
                     suffix: str = None,
                     append: bool = True,
                     memory: str = None,
                     slurm_account: str = None) -> str:

    if not _is_slurm_available():
        raise RuntimeError("SLURM is not available. Please check your environment.")

    cmd = f'job_{data_type}_{idx+1}=$(sbatch ' \

    if idx > 0:
        cmd += f'--dependency=afterok:$job_{data_type}_{idx} '

    if slurm_account:
        cmd += f'--account={slurm_account} '

    if memory:
        cmd += f'--mem={memory} '

    cmd += f'--ntasks={ntasks} '
    cmd += '--time=02:00:00 '
    cmd += f'--job-name="gen_ocelot_{data_type}_{idx+1}" '
    cmd += f'--wrap="srun -n {ntasks} python {reader_path} {start.strftime("%Y-%m-%d")} {end.strftime("%Y-%m-%d")} {data_type} {output_type} '

    if suffix:
        cmd += f'-s {suffix} '

    if append:
        cmd += '-a '

    cmd += '" | awk \'{print $4}\')'

    return cmd


def _make_parallel_cmd(reader_path: str,
                       start: datetime,
                       end: datetime,
                       ntasks: int,
                       data_type: str,
                       output_type: str,
                       suffix: str = None,
                       append: bool = True) -> str:

    if _is_slurm_available():
        cmd = f'srun -n {ntasks} python {reader_path} {start.strftime("%Y-%m-%d")} {end.strftime("%Y-%m-%d")} {data_type} {output_type} '
        if suffix:
            cmd += f'-s {suffix} '

        if append:
            cmd += '-a '

    else:
        cmd = f'mpirun -n {ntasks} python {reader_path} {start.strftime("%Y-%m-%d")} {end.strftime("%Y-%m-%d")} {data_type} {output_type} '
        if suffix:
            cmd += f'-s {suffix} '

        if append:
            cmd += '-a '

    return cmd


def _make_serial_cmd(reader_path: str,
                     start: datetime,
                     end: datetime,
                     data_type: str,
                     output_type: str,
                     suffix: str = None,
                     append=True) -> str:

    cmd = f'python {reader_path} {start.strftime("%Y-%m-%d")} {end.strftime("%Y-%m-%d")} {data_type} {output_type} '

    if suffix:
        cmd += f'-s {suffix} '

    if append:
        cmd += '-a '

    return cmd


def _split_datetime_range(start: datetime, end: datetime, num_days: int) -> list:
    """
    Split the datetime range into chunks of num_days days.
    """
    delta = end - start
    num_chunks = (delta.days + num_days - 1) // num_days

    ranges = []
    for i in range(num_chunks):
        chunk_start = start + i * timedelta(days=num_days)
        chunk_end = min(end, chunk_start + timedelta(days=num_days - 1))
        ranges.append((chunk_start, chunk_end))

    return ranges


def _batch_gen(reader_path: str,
               start: datetime,
               end: datetime,
               max_days: int,
               ntasks: int,
               gen_type: str,
               output_type: str,
               suffix: str = None,
               append: bool = True,
               memory: str = None,
               slurm_account: str = None) -> None:
    ranges = _split_datetime_range(start, end, max_days)
    cmds = []
    for idx, (start, end) in enumerate(ranges):
        if idx == 0:
            cmds.append(
                _make_sbatch_cmd(reader_path,
                                 idx,
                                 start,
                                 end,
                                 ntasks,
                                 gen_type,
                                 output_type,
                                 suffix=suffix,
                                 append=append,
                                 memory=memory,
                                 slurm_account=slurm_account))
        else:
            cmds.append(
                _make_sbatch_cmd(reader_path,
                                 idx,
                                 start,
                                 end,
                                 ntasks,
                                 gen_type,
                                 output_type,
                                 suffix=suffix,
                                 memory=memory,
                                 slurm_account=slurm_account))

    cmd = '\n'.join(cmds)
    print(cmd)
    os.system(cmd)


def _parallel_gen(reader_path: str,
                  start: datetime,
                  end: datetime,
                  ntasks: int,
                  gen_type: str,
                  output_type: str,
                  suffix: str = None,
                  append: bool = True) -> None:
    cmd = _make_parallel_cmd(reader_path, start, end, ntasks, gen_type, output_type, suffix=suffix, append=append)
    print(cmd)
    os.system(cmd)


def _serial_gen(reader_path: str,
                start: datetime,
                end: datetime,
                gen_type: str,
                output_type: str,
                suffix: str = None,
                append: bool = True) -> None:
    cmd = _make_serial_cmd(reader_path, start, end, gen_type, output_type, suffix=suffix, append=append)
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    from config import Config
    config = Config()

    data_types = config.get_data_type_names()
    choices = ['all'] + data_types

    parser = argparse.ArgumentParser()
    parser.add_argument('start_date', help='Start date in YYYY-MM-DD format')
    parser.add_argument('end_date', help='End date in YYYY-MM-DD format')
    parser.add_argument('type', choices=choices, help='Data type to generate. "all" generates all data types.')
    parser.add_argument("output_type", choices=['zarr', 'parquet'], help='Output file type')
    parser.add_argument('-s', '--suffix', required=False, help='Suffix for the output file(s)')
    parser.add_argument('-p', '--parallel', action='store_true', help='Run in parallel (using either srun or mpirun).')
    parser.add_argument('-b', '--batch', action='store_true', help='Run in batch mode (using sbatch). Chunks the data into multiple tasks if needed.')
    parser.add_argument('-a', '--append', action='store_true', help='Append to existing data')
    parser.add_argument('--slurm_account', required=False, help='SLURM account name for batch jobs')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    def call_generator(gen_type):
        type_config = config.get_data_type(gen_type)
        reader_path = get_reader_path(type_config.type)
        if args.batch:
            _batch_gen(reader_path,
                       start_date,
                       end_date,
                       type_config.batch_days,
                       type_config.num_tasks,
                       gen_type,
                       args.output_type,
                       suffix=args.suffix,
                       append=args.append,
                       memory=type_config.memory,
                       slurm_account=args.slurm_account)
        elif args.parallel:
            _parallel_gen(reader_path,
                          start_date,
                          end_date,
                          type_config.num_tasks,
                          gen_type,
                          args.output_type,
                          suffix=args.suffix,
                          append=args.append)
        else:
            _serial_gen(reader_path,
                        start_date,
                        end_date,
                        gen_type,
                        args.output_type,
                        suffix=args.suffix,
                        append=args.append)

    if args.type == 'all':
        for gen_type in data_types:
            call_generator(gen_type)
    else:
        call_generator(args.type)
