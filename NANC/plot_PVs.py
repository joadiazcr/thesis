import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


def preprocess_data(datafile):
    df = pd.read_csv(datafile, sep='\t')
    df = df.drop(['Unnamed: 9'], axis=1)
    df = df.replace("BadVal", 0.0)
    df['Time']= pd.to_datetime(df['Time'])
    df = df.astype({'ACCL:L3B:1610:PZT:AVDIFF:MEAN []':'float'})
    df = df.astype({'ACCL:L3B:1610:PZT:BVDIFF:MEAN []':'float'})
    df = df.astype({'ACCL:L3B:1610:PZT:DF:STD []':'float'})
    df = df.astype({'ACCL:L3B:1610:DF [Hz]':'float'})
    df = df.astype({'ACCL:L3B:1610:CAV:ASTD []':'float'})
    df = df.astype({'ACCL:L3B:1610:DFBEST [Hz]':'float'})
    df = df.astype({'ACCL:L3B:1610:CAV:PSTD []':'float'})
    df = df.astype({'ACCL:L3B:1610:CAV:ARSTD []':'float'})
    
    pref = df.columns[1]
    pref = pref.split(':')
    pref = ':'.join(pref[0:3])

    return df, pref


def plot_data(df, pref):
    print(df.dtypes)
    tt = "Cavity detuning\n"+pref
    t = df.Time
    df_best = df.loc[:,pref+":DFBEST [Hz]"]

    #Important moments
    data_start = datetime.strptime("05/17/2022 17:42:16.084670", "%m/%d/%Y %H:%M:%S.%f")
    data_end = datetime.strptime("05/17/2022 19:34:37.781695", "%m/%d/%Y %H:%M:%S.%f")

    fault_start = datetime.strptime("05/17/2022 17:47:54.282472", "%m/%d/%Y %H:%M:%S.%f")
    fault_end = datetime.strptime("05/17/2022 18:05:54.003660", "%m/%d/%Y %H:%M:%S.%f")
    fault_start_2 = datetime.strptime("05/17/2022 18:15:46.861792", "%m/%d/%Y %H:%M:%S.%f")
    fault_end_2 = datetime.strptime("05/17/2022 18:18:09.823903", "%m/%d/%Y %H:%M:%S.%f")
    fault_start_3 = datetime.strptime("05/17/2022 18:37:10.528524", "%m/%d/%Y %H:%M:%S.%f")
    fault_end_3 = datetime.strptime("05/17/2022 18:43:31.446773", "%m/%d/%Y %H:%M:%S.%f")
    fault_start_4 = datetime.strptime("05/17/2022 19:06:41.102990", "%m/%d/%Y %H:%M:%S.%f")
    fault_end_4 = datetime.strptime("05/17/2022 19:18:22.353620", "%m/%d/%Y %H:%M:%S.%f")

    config_change = datetime.strptime("05/17/2022 18:13:15.00000", "%m/%d/%Y %H:%M:%S.%f")
    config_change_2 = datetime.strptime("05/17/2022 18:43:31.446773", "%m/%d/%Y %H:%M:%S.%f")
    config_change_3 = datetime.strptime("05/17/2022 18:53:00.00", "%m/%d/%Y %H:%M:%S.%f")
    config_change_4 = datetime.strptime("05/17/2022 18:56:00.00", "%m/%d/%Y %H:%M:%S.%f")
    config_change_5 = datetime.strptime("05/17/2022 19:18:22.353620", "%m/%d/%Y %H:%M:%S.%f")
    config_change_6 = datetime.strptime("05/17/2022 19:29:00.00", "%m/%d/%Y %H:%M:%S.%f")


    # Microphonics data acquisition
    micro_acq_1 = datetime.strptime("05/17/2022 18:28:23.00000", "%m/%d/%Y %H:%M:%S.%f")
    micro_acq_2 = datetime.strptime("05/17/2022 18:48:36.00000", "%m/%d/%Y %H:%M:%S.%f")
    micro_acq_3 = datetime.strptime("05/17/2022 18:53:43.00000", "%m/%d/%Y %H:%M:%S.%f")
    micro_acq_4 = datetime.strptime("05/17/2022 19:03:53.00000", "%m/%d/%Y %H:%M:%S.%f")
    micro_acq_5 = datetime.strptime("05/17/2022 19:27:17.00000", "%m/%d/%Y %H:%M:%S.%f")
    micro_acq_6 = datetime.strptime("05/17/2022 19:30:50.00000", "%m/%d/%Y %H:%M:%S.%f")

    micro_acq = [micro_acq_1, micro_acq_2, micro_acq_3, micro_acq_4, micro_acq_5, micro_acq_6]


    ax = df.plot(x='Time', y=pref+":DFBEST [Hz]", xlim=(data_start, data_end))
   
    ax.scatter(micro_acq, np.ones(len(micro_acq))*50, marker='*', color='orange', label='Microphonics data acquisition')

    plt.text(fault_end, 50, 'SEL\nFDBK OFF\nARC OFF')
    ax.axvline(config_change, color="green", linestyle="--", label='Configuration Change')
    plt.text(config_change, 50, 'SEL\nFDBK OFF\nARC ON')
    ax.axvline(config_change_2, color="green", linestyle="--")
    plt.text(config_change_2, 50, 'SEL\nFDBK OFF\nARC OFF')
    ax.axvline(config_change_3, color="green", linestyle="--")
    plt.text(config_change_3, 50, 'SEL\nFDBK ON\nARC OFF')
    ax.axvline(config_change_4, color="green", linestyle="--")
    plt.text(config_change_4, 50, 'SEL\nFDBK ON\nARC ON')
    ax.axvline(config_change_5, color="green", linestyle="--")
    plt.text(config_change_5, 50, 'SELAP\nFDBK ON\nARC ON')
    ax.axvline(config_change_6, color="green", linestyle="--")
    plt.text(config_change_6, 50, 'SELAP\nFDBK ON\nARC OFF')



    ax.axvspan(fault_start, fault_end, alpha=0.5, color='red', label='Fault. RF OFF')
    ax.axvspan(fault_start_2, fault_end_2, alpha=0.5, color='red')
    ax.axvspan(fault_start_3, fault_end_3, alpha=0.5, color='red')
    ax.axvspan(fault_start_4, fault_end_4, alpha=0.5, color='red')

    plt.legend()
    plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Plot data acquired with with the StripTool")
    parser.add_argument('-f', '--file', dest='datafile', help='StripTool data file')

    args = parser.parse_args()

    df, pref = preprocess_data(args.datafile)
    plot_data(df, pref)
