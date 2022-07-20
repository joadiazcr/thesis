import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from datetime import timedelta


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def preprocess_data(datafile, pv):
    df = pd.read_csv(datafile, sep='\t')
    try:
        df = df.drop(['Unnamed: 9'], axis=1)
    except KeyError:
        print("No Unnamed column")

    df = df.replace("BadVal", 0.0)
    df['Time'] = pd.to_datetime(df['Time'])

    pref = df.columns[1]
    pref = pref.split(':')
    pref = ':'.join(pref[0:3]) + ':'

    df = df.astype({pref + pv: 'float'})

#    df = df.astype({'ACCL:L3B:1610:PZT:AVDIFF:MEAN []': 'float'})
#    df = df.astype({'ACCL:L3B:1610:PZT:BVDIFF:MEAN []': 'float'})
#    df = df.astype({'ACCL:L3B:1610:PZT:DF:STD []': 'float'})
#    df = df.astype({'ACCL:L3B:1610:DF [Hz]': 'float'})
#    df = df.astype({'ACCL:L3B:1610:CAV:ASTD []': 'float'})
#    df = df.astype({'ACCL:L3B:1610:DFBEST [Hz]': 'float'})
#    df = df.astype({'ACCL:L3B:1610:CAV:PSTD []': 'float'})
#    df = df.astype({'ACCL:L3B:1610:CAV:ARSTD []': 'float'})

    return df, pref


def plot_data(df, pref, pv):
    d_fmt = "%m/%d/%Y %H:%M:%S.%f"
    fault_start = []
    fault_end = []
    config_change = []
    configs = []
    micro_acq = []

    # Important moments CM16 C1 2022-May-17
    if '1610' in pref:
       data_start = datetime.strptime("05/17/2022 17:42:16.084670", d_fmt)
       data_end = datetime.strptime("05/17/2022 19:34:37.781695", d_fmt)

       fault_start.append(datetime.strptime("05/17/2022 17:47:54.282472", d_fmt))
       fault_end.append(datetime.strptime("05/17/2022 18:05:54.003660", d_fmt))
       fault_start.append(datetime.strptime("05/17/2022 18:15:46.861792", d_fmt))
       fault_end.append(datetime.strptime("05/17/2022 18:18:09.823903", d_fmt))
       fault_start.append(datetime.strptime("05/17/2022 18:37:10.528524", d_fmt))
       fault_end.append(datetime.strptime("05/17/2022 18:43:31.446773", d_fmt))
       fault_start.append(datetime.strptime("05/17/2022 19:06:41.102990", d_fmt))
       fault_end.append(datetime.strptime("05/17/2022 19:18:22.353620", d_fmt))

       d = timedelta(minutes=7)
       config_change.append(datetime.strptime("05/17/2022 18:13:15.00000", d_fmt))
       config_change.append(datetime.strptime("05/17/2022 18:43:31.446773", d_fmt))
       config_change.append(datetime.strptime("05/17/2022 18:53:00.00", d_fmt))
       config_change.append(datetime.strptime("05/17/2022 18:56:00.00", d_fmt))
       config_change.append(datetime.strptime("05/17/2022 19:18:22.353620", d_fmt))
       config_change.append(datetime.strptime("05/17/2022 19:29:00.00", d_fmt))
       configs = ['SEL\nFDBK OFF\nARC OFF',
                  'SEL\nFDBK OFF\nARC ON',
                  'SEL\nFDBK OFF\nARC OFF',
                  'SEL\nFDBK ON\nARC OFF',
                  'SEL\nFDBK ON\nARC ON',
                  'SELAP\nFDBK ON\nARC ON',
                  'SELAP\nFDBK ON\nARC OFF']

       micro_acq.append(datetime.strptime("05/17/2022 18:28:23.00000", d_fmt))
       micro_acq.append(datetime.strptime("05/17/2022 18:48:36.00000", d_fmt))
       micro_acq.append(datetime.strptime("05/17/2022 18:53:43.00000", d_fmt))
       micro_acq.append(datetime.strptime("05/17/2022 19:03:53.00000", d_fmt))
       micro_acq.append(datetime.strptime("05/17/2022 19:27:17.00000", d_fmt))
       micro_acq.append(datetime.strptime("05/17/2022 19:30:50.00000", d_fmt))

       delta_y = 50

    # Important moments CM16 C4 2022-May-17
    elif '1640' in pref:
       data_start = datetime.strptime("05/17/2022 19:59:16.946879", d_fmt)
       data_end = datetime.strptime("05/17/2022 20:39:15.132822", d_fmt)

       fault_start.append(datetime.strptime("05/17/2022 20:06:47.688332", d_fmt))
       fault_end.append(datetime.strptime("05/17/2022 20:12:21.544458", d_fmt))

       d = timedelta(minutes=6)
       config_change.append(datetime.strptime("05/17/2022 20:19:00.00", d_fmt))
       config_change.append(datetime.strptime("05/17/2022 20:27:00.00", d_fmt))
       configs = ['SELAP\nFDBK ON\nARC OFF',
                  'SELAP\nFDBK ON\nARC ON',
                  'SELAP\nFDBK ON\nARC OFF']
 
       micro_acq.append(datetime.strptime("05/17/2022 20:24:09.00", d_fmt))
       micro_acq.append(datetime.strptime("05/17/2022 20:32:20.00", d_fmt))

       delta_y = 20

    ax = df.plot(x='Time', y=pref+pv,
                 xlim=(data_start, data_end))

    ax.scatter(micro_acq, np.ones(len(micro_acq))*delta_y, marker='*',
               color='orange', label='Microphonics data acquisition')

    for i, c in enumerate(configs):
        if i == 0:
            plt.text(config_change[i]-d, delta_y, c)
        else:
            plt.text(config_change[i-1], delta_y, c)

    for i, c_c in enumerate(config_change):
        if i == 0:
            ax.axvline(c_c, color="green", linestyle="--",
                       label='Configuration Change')
        else:
            ax.axvline(c_c, color="green", linestyle="--")

    for fault in range(0,len(fault_start)):
        if fault == 0:
            ax.axvspan(fault_start[fault], fault_end[fault], 
                       alpha=0.5, color='red',
                       label='Fault. RF OFF')
        else:
            ax.axvspan(fault_start[fault], fault_end[fault], alpha=0.5, color='red')

    plt.legend()
    plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Plot data acquired with \
                            the StripTool")
    parser.add_argument('-f', '--file', dest='datafile', required=True,
                        help='StripTool data file')
    parser.add_argument('-p', '--pv', dest='pv', default='DFBEST [Hz]',
                        help='StripTool data file')


    args = parser.parse_args()

    df, pref = preprocess_data(args.datafile, args.pv)
    plot_data(df, pref, args.pv)
