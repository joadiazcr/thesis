import numpy as np
from pathlib import Path
import sys


def stack_res_files(file, output_file, split_index):
    data = np.loadtxt(file)
    s, c = data.shape
    print(f'Original shape is {s} X {c}')

    part1 = data[:split_index, :]
    part2 = data[-split_index:, :]

    new_array = np.hstack((part1, part2))

    np.savetxt(output_file, new_array, fmt='%.6f')


if __name__ == "__main__":

    from argparse import ArgumentParser

    desc = 'Script to stack files created by res_data_acq.py.'
    parser = ArgumentParser(description=desc)
    parser.add_argument("-f", "--file", dest='datafile', required=True,
                        help="Microhponics file colected with res_data_acq.py")
    parser.add_argument("-o", "--output_file", required=True,
                        help="Path and name for output file")
    parser.add_argument("-s", "--split", type=int, required=True,
                        help="Index of the row to split the data")
    args = parser.parse_args()

    stack_res_files(args.datafile, args.output_file, args.split)
