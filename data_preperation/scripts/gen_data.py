import sys
import os
import argparse
from datetime import datetime

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

    cmd += f'--ntasks={ntasks} \
             --job-name="gen_ocelot_{type}_{idx+1}" \
             --wrap="python {runner_path} {start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')} {type} '

    if output_name:
        cmd += f'-o {output_name} '

    if not append:
        cmd += '-a False '

    cmd += '" | awk \'{print $4}\')'

    return cmd

def _split_datetime_range(start:datetime, end:datetime, num_days:int):
    """
    Split the datetime range into smaller ranges of num_days days each.
    """
    delta = (end - start) / num_days
    ranges = []
    for i in range(num_days):
        new_start = start + i * delta
        new_end = start + (i + 1) * delta
        ranges.append((new_start, new_end))
    return ranges

def _gen(start : datetime, end :datetime, max_days, ntasks, gen_type:str, output_name:str=None, append=True):
    ranges = _split_datetime_range(start, end, max_days)

    for idx, (start, end) in enumerate(ranges):
        cmd = _make_sbatch_cmd(idx, start, end, ntasks, gen_type)
        print(cmd)
        os.system(cmd)

def _gen_atms(start : datetime, end :datetime):
    _gen(start, end, 15, 24, 'atms')

def _gen_radiosonde(start : datetime, end :datetime):
    _gen(start, end, 30, 4, 'radiosonde')

def _gen_surface_pressure(start : datetime, end :datetime):
    _gen(start, end, 30, 4, 'surface_pressure')

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
    parser.add_argument('-a', '--append', required=False, default=True)

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    def call_generator(gen_type):
        gen_args = (start_date, end_date,)
        if args.output_name:
            gen_args += (args.output_name,)
        if args.append:
            gen_args += (args.append,)

        gen_dict[gen_type](*gen_args)

    if args.type == 'all':
        for gen_type in gen_dict.keys():
            call_generator(gen_type)
    else:
        call_generator(args.type)
