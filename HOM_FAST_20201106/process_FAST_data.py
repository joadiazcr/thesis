import numpy as np
import matplotlib.pyplot as plt
import argparse
from os.path import dirname, join as pjoin
import scipy.io as sio


def plot_raw_HOMs(data):
    HOMs = data['HOMs']
    num_HOMs = HOMs.shape[0]
    reps = HOMs.shape[1]
    points = HOMs.shape[2]
    print('Num HOMs = ', num_HOMs)
    print('Repetitions = ', reps)
    print('Points = ', points)
    for hom in range(0,num_HOMs):
        plt.plot(HOMs[hom][0], label='HOM %s' %(hom+1))
    plt.xlim([740,790])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    des = 'Procces and plot FAST data'
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('-f', '--file', dest='data_file', required=True,
                        help='File with data to be processed/plotted')

    args = parser.parse_args()

    data = sio.loadmat(args.data_file)
    plot_raw_HOMs(data)
