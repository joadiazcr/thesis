from dataclasses import dataclass, field
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    print(ca_file.df.head(20))

    df_wf_raw = ca_file.df.loc['CAV:IWF', 'Value']
    df_wf = np.concatenate(df_wf_raw).tolist()
    print(len(df_wf))
    print(df_wf[:20])
    plt.plot(df_wf)
    plt.show()
