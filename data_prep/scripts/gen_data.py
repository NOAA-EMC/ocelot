import sys
import os
import argparse
from datetime import datetime, timedelta

base_path = os.path.split(os.path.realpath(__file__))[0]
runner_path = os.path.realpath(os.path.join(base_path, '../src/gen_model_data.py'))

sys.path.append(os.path.realpath(os.path.join(base_path, '..', 'src')))

def _make_sbatch_cmd(idx:int,
                     start:datetime,
                     end:datetime,
                     ntasks:int,
                     type:str,
                     suffix:str=None,
                     append=True):

    cmd = f'job_{idx+1}=$(sbatch ' \

    if idx > 0:
        cmd += f'--dependency=afterok:$job_{idx} '

    cmd += f'--ntasks={ntasks} '
    cmd += f'--time=02:00:00 '
    cmd += f'--job-name="gen_ocelot_{type}_{idx+1}" '
    cmd += f'--wrap="srun -n{ntasks} python {runner_path} {start.strftime("%Y-%m-%d")} {end.strftime("%Y-%m-%d")} {type} '

    if suffix:
        cmd += f'-s {suffix} '

    if not append:
        cmd += '--append False '

    cmd += '" | awk \'{print $4}\')'

    return cmd

def _split_datetime_range(start:datetime, end:datetime, num_days:int):
    """
    Split the datetime range into chunks of num_days days.
    """
    delta = end - start
    num_chunks = delta.days // num_days + (1 if delta.days % num_days > 0 else 0)
    num_chunks = max(1, num_chunks)

    ranges = []
    for i in range(num_chunks):
        chunk_start = start + i * timedelta(days=num_days)
        chunk_end = min(end, chunk_start + timedelta(days=num_days))
        ranges.append((chunk_start, chunk_end))

    return ranges

def _gen(start : datetime, end :datetime, max_days, ntasks, gen_type:str, suffix:str=None):
    ranges = _split_datetime_range(start, end, max_days)

    cmds = []
    for idx, (start, end) in enumerate(ranges):
        if idx == 0:
            cmds.append(_make_sbatch_cmd(idx, start, end, ntasks, gen_type, suffix=suffix, append=False))
        else:
            cmds.append(_make_sbatch_cmd(idx, start, end, ntasks, gen_type, suffix=suffix))

    cmd = '\n'.join(cmds)
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    from data_reader import Config
    config = Config()

    data_types = config.get_data_type_names()
    choices = ['all'] + data_types

    parser = argparse.ArgumentParser()
    parser.add_argument('start_date')
    parser.add_argument('end_date')
    parser.add_argument('type', choices=choices)
    parser.add_argument('-s', '--suffix', required=False)

    args = parser.parse_args()

    type_config = config.get_data_type(args.type)
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    def call_generator(gen_type):
        _gen(start_date,
             end_date,
             type_config.batch_days,
             type_config.num_tasks,
             type_config.name,
             suffix=args.suffix)

    if args.type == 'all':
        for gen_type in data_types:
            call_generator(gen_type)
    else:
        call_generator(args.type)
