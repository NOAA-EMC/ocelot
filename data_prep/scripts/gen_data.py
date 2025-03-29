import sys
import os
import argparse
from datetime import datetime, timedelta

base_path = os.path.split(os.path.realpath(__file__))[0]
runner_path = os.path.realpath(os.path.join(base_path, '../src/gen_model_data.py'))

def _make_sbatch_cmd(idx:int,
                     start:datetime,
                     end:datetime,
                     ntasks:int,
                     type:str,
                     output_name:str=None,
                     append=True):

    cmd = f'job_{idx+1}=$(sbatch ' \

    if idx > 0:
        cmd += f'--dependency=afterok:$job_{idx} '

    cmd += f'--ntasks={ntasks} '
    cmd += f'--time=02:00:00 '
    cmd += f'--job-name="gen_ocelot_{type}_{idx+1}" '
    cmd += f'--wrap="python {runner_path} {start.strftime("%Y-%m-%d")} {end.strftime("%Y-%m-%d")} {type} '

    if output_name:
        cmd += f'-o {output_name} '

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

def _gen(start : datetime, end :datetime, max_days, ntasks, gen_type:str, output_name:str=None):
    ranges = _split_datetime_range(start, end, max_days)

    cmds = []
    for idx, (start, end) in enumerate(ranges):
        if idx == 0:
            cmds.append(_make_sbatch_cmd(idx, start, end, ntasks, gen_type, output_name=output_name, append=False))
        else:
            cmds.append(_make_sbatch_cmd(idx, start, end, ntasks, gen_type, output_name=output_name))

    cmd = '\n'.join(cmds)
    print(cmd)
    os.system(cmd)

def _gen_atms(start : datetime, end :datetime, output_name:str=None):
    _gen(start, end, 15, 24, 'atms', output_name=output_name)

def _gen_radiosonde(start : datetime, end :datetime, output_name:str=None, append=True):
    _gen(start, end, 30, 4, 'radiosonde', output_name=output_name)

def _gen_surface_pressure(start : datetime, end :datetime, output_name:str=None, append=True):
    _gen(start, end, 30, 4, 'surface_pressure', output_name=output_name)

if __name__ == "__main__":
    gen_dict = {
        'atms': _gen_atms,
        'radiosonde': _gen_radiosonde,
        'surface_pressure': _gen_surface_pressure
    }

    choices = ['all'] + list(gen_dict.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument('start_date')
    parser.add_argument('end_date')
    parser.add_argument('type', choices=choices)
    parser.add_argument('-o', '--output_name', required=False)

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    def call_generator(gen_type):
        gen_args = (start_date, end_date,)
        gen_kwargs = {}
        if args.output_name:
            gen_kwargs['output_name'] = args.output_name

        gen_dict[gen_type](*gen_args, **gen_kwargs)

    if args.type == 'all':
        for gen_type in gen_dict.keys():
            call_generator(gen_type)
    else:
        call_generator(args.type)
