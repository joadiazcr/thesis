from dataclasses import dataclass, field
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_directory)
import utils


def flatten_lists(x):
    shape = np.shape(x)
    if len(shape) > 1:
        return list(itertools.chain.from_iterable(x))
    else:
        return x


@dataclass
class CamonitorFile():
    filename: str
    prefix: str
    df: pd.DataFrame = field(default=None, repr=False)
    pv_dict: dict = field(default_factory=dict, repr=False)

    def load_data(self):
        with open(self.filename, 'r') as file:
            for line in file:
                split_line = line.strip().split()
                if split_line and (self.prefix in split_line[0]):
                    pv = ':'.join(split_line[0].split(':')[3:])
                    if pv not in self.pv_dict:
                        self.pv_dict[pv] = []
                    if split_line[0].split(':')[-1] not in ['WF', 'IWF', 'QWF']:
                        try:
                            self.pv_dict[pv].append(float(split_line[3]))
                        except ValueError:
                            self.pv_dict[pv].append(split_line[3])
                    else:
                        float_list = [float(item) for item in split_line[4:]]
                        self.pv_dict[pv].append(float_list)

        self.df = pd.DataFrame(self.pv_dict.items(), columns=['PV', 'Value'])
        self.df.set_index('PV', inplace=True)
        self.df['Value'] = self.df['Value'].apply(flatten_lists)

    def time_axis(self):
        decim_rf = self.df.loc["ACQ_DECIM", 'Value'][0]
        self.ts_rf = (14 * 2 * 33 * decim_rf) / 1.315e9
        self.taxis_rf = np.arange(len(self.df.loc[f"DF:WF", 'Value'])) * self.ts_rf

        decim_rc = 1 # Default. Check!
        self.ts_rc = decim_rc / 2.0e3 # 2 KHz is the maximun sampling rate
        self.taxis_rc = np.arange(len(self.df.loc[f"PZT:DF:WF", 'Value'])) * self.ts_rc

    def plot_wf(self, pv):
        i_wf = self.df.loc[f"{pv}:IWF", 'Value']
        q_wf = self.df.loc[f"{pv}:QWF", 'Value']
        wf = np.array(i_wf) + 1j * np.array(q_wf)
        fig, ax = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
        ax[0].plot(self.taxis_rf, np.abs(wf))
        ax[0].axhline(y=self.df.loc[f"ADES", 'Value'][0], color='r', linestyle='--', label='ADES')
        ax[1].plot(self.taxis_rf, np.degrees(np.angle(wf)))
        ax[2].plot(self.taxis_rf, i_wf)
        ax[3].plot(self.taxis_rf, q_wf)
        ax[0].set_xlim(0, self.taxis_rf[-1])
        ax[0].set_ylabel('Amplitude [MV]')
        ax[1].set_ylabel('Phase [rad]')
        ax[2].set_ylabel('I [MV]')
        ax[3].set_ylabel('Q [MV]')
        ax[3].set_xlabel('Time [s]')
        ax[0].legend()
        plt.show()

    def plot_rfs_res_detuning(self):
        det_wf = self.df.loc["DF:WF", 'Value']
        det_pzt_wf = self.df.loc["PZT:DF:WF", 'Value']
        time_axs = [self.taxis_rc, self.taxis_rf]
        data = [det_pzt_wf, det_wf]
        labels = ['PZT Detuning', 'RFS Detuning']
        dts = [self.ts_rc, self.ts_rf]
        utils.plot_detuning(time_axs, data, labels, dts, max_freq=250)


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Archive data plotter")
    parser.add_argument('-f', '--file', dest='datafile', required=True,
                        help='Waveform data file')
    parser.add_argument('-p', '--prefix', dest='prefix', required=True,
                        help='PV cavity prefix (e.g. VTS:L1B:H110:)')
    args = parser.parse_args()

    print("Changing font settings...")
    plt.rc('font', size=18)
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    plt.rc('legend', fontsize=14)    # legend fontsize
    plt.rc('figure', titlesize=18)  # fontsize of the figure title

    ca_file = CamonitorFile(args.datafile, args.prefix)
    ca_file.load_data()
    ca_file.time_axis()
    print(ca_file.df.head(40))

    #ca_file.plot_wf('CAV')
    ca_file.plot_rfs_res_detuning()
