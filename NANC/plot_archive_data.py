import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.fft import fftfreq
from scipy import signal
import pandas as pd


#plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def archive_plot(data_f):
    df_fwd_amean = pd.read_csv('~/Documents/NANC/shift_20251106/shift_20251106_amean',
                     sep='\t', comment='#', index_col=False,  parse_dates=['Timestamp'])
    df_ades = pd.read_csv('~/Documents/NANC/shift_20251106/shift_20251106_ades',
                     sep='\t', comment='#', index_col=False,  parse_dates=['Timestamp'])
    df_cav_amean = pd.read_csv('~/Documents/NANC/shift_20251106/shift_20251106_aactmean',
                     sep='\t', comment='#', index_col=False,  parse_dates=['Timestamp'])
    df_df_std = pd.read_csv('~/Documents/NANC/shift_20251106/shift_20251106_df_std',
                     sep='\t', comment='#', index_col=False,  parse_dates=['Timestamp'])
    df_clip_cnt = pd.read_csv('~/Documents/NANC/shift_20251106/shift_20251106_fb_sum_cnt',
                     sep='\t', comment='#', index_col=False,  parse_dates=['Timestamp'])

    t_fwd_amean = df_fwd_amean['Timestamp'].to_numpy()
    y_fwd_amean = df_fwd_amean['ACCL:L3B:1920:FWD:AMEAN'].to_numpy()
    t_ades = df_ades['Timestamp'].to_numpy()
    y_ades = df_ades['ACCL:L3B:1920:ADES'].to_numpy()
    t_cav_amean = df_cav_amean['Timestamp'].to_numpy()
    y_cav_amean = df_cav_amean['ACCL:L3B:1920:AACTMEAN'].to_numpy()
    t_df_std = df_df_std['Timestamp'].to_numpy()
    y_df_std = df_df_std['ACCL:L3B:1920:DF:STD'].to_numpy()
    t_clip_cnt = df_clip_cnt['Timestamp'].to_numpy()
    y_clip_cnt = df_clip_cnt['ACCL:L3B:1920:FB_SUM_CNTT'].to_numpy()

    fig, ax = plt.subplots()
    lns1 = ax.plot(t_cav_amean, y_cav_amean, label='AACTMEAN',
            color='blue')
    lns2 = ax.step(t_ades, y_ades, label='ADES', where='post',
            color='orange')
    
    ax2 = ax.twinx()      # create the twin axis
    ax2.plot(t_fwd_amean, y_fwd_amean, color="red", label="FWD:AMEAN")
    ax2.spines["left"].set_position(("axes", -0.08))  # shift left
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.set_ticks_position('left')

    ax3 = ax.twinx()      # create the twin axis
    ax3.plot(t_df_std, y_df_std, label="DF:STD")

    ax4 = ax.twinx()      # create the twin axis
    ax4.step(t_clip_cnt, y_clip_cnt, label="FB_SUM_CNTT")
    ax4.spines["right"].set_position(("axes", 1.08))  # shift left

    ax.set_xlabel('Time', fontsize=24)
    ax.set_ylabel('sqrt(W)', fontsize=24)
    plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Archive data plotter")
    parser.add_argument('-f', '--file', dest='datafile', required=True,
                        help='Waveform data file')
    parser.add_argument('-t', '--title', dest='tt', help='Plot Title',
                        default='')
    args = parser.parse_args()

    if args.tt == '':
        print("Changing font settings...")
        plt.rc('font', size=18)
        plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
        plt.rc('legend', fontsize=14)    # legend fontsize
        plt.rc('figure', titlesize=18)  # fontsize of the figure title

    archive_plot(args.datafile)