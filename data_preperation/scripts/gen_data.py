import sys
import os
import argparse
from datetime import datetime

base_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.realpath(os.path.join(base_path, '../src')))

from gen_model_data import create_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('start_date')
    parser.add_argument('end_date')
    parser.add_argument('type')
    parser.add_argument('-o', '--output_name', required=False)
    parser.add_argument('-a', '--append', required=False, default=True)

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    create_data(start_date, end_date, args.type, args.output_name, args.append)

