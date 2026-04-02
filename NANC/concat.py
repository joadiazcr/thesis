import numpy as np
from pathlib import Path
import sys


def stack_res_files(file_list, output_file):
    arrays = []
    for f in file_list:
        path = Path(f)
        if path.is_file():
            data = np.loadtxt(f)
            print(f'{data.shape} {f}')
            arrays.append(data)
        else:
            sys.exit(f'Error: {f} does not exist!')
            exit()

    data_total = np.column_stack(arrays)
    np.savetxt(output_file, data_total, fmt="%.6f")
    print(f'\n{data_total.shape} {output_file}')


if __name__ == "__main__":

    from argparse import ArgumentParser

    desc = 'Script to stack files created by res_data_acq.py.'
    parser = ArgumentParser(description=desc)
    parser.add_argument("-f", "--file_list", nargs="+", required=True,
                        help="List of files to stack")
    parser.add_argument("-o", "--output_file", required=True,
                        help="Path and name for output file")
    args = parser.parse_args()

    stack_res_files(args.file_list, args.output_file)
