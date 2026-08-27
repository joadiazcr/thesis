from dataclasses import dataclass, field
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import matplotlib.dates as mdates
import datetime
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_directory)
import utils
import plot_camonitor


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

    def split_plot_rfs_det(self, split_index):
        det_wf = self.df.loc["DF:WF", 'Value']
        time_axs = [self.taxis_rf]
        det_wf_a = det_wf[:split_index]
        det_wf_b = det_wf[-split_index:]
        data = [det_wf_a, det_wf_b]
        labels = ['NANC OFF', 'NANC ON']
        time_axs = [self.taxis_rf[:split_index], self.taxis_rf[:split_index]]
        dts = [self.ts_rf, self.ts_rf]
        utils.plot_detuning(time_axs, data, labels, dts, max_freq=250, req=True)

    def plot_overview(self):
        target_pvs = ['AACTMEAN', 'DF:STD', 'CNTT', 'FWD:AMEAN']
        d_fmt = "%m/%d/%Y %H:%M:%S.%f"

        ca_overview = plot_camonitor.CamonitorFile(self.filename)
        ca_overview.load_data()
        plot_df = ca_overview.df

        # Filter for specific PVs if requested
        if target_pvs:
            mask = plot_df['variable'].str.contains('|'.join(target_pvs))
            plot_df = plot_df[mask]
        
        # Get unique variables to plot
        groups = plot_df.groupby('variable')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        color = 'black'
        for name, group in groups:
            if 'CNTT' in name:
                ax4 = ax.twinx()
                lns1 = ax4.step(group['timestamp'], group['value_numeric'],
                                label=':'.join(name.split(':')[3:]),
                                color='brown', linewidth=4, alpha=0.8)
                ax4.spines["right"].set_position(("axes", 1.05))
                ax4.set_ylabel('Clip limit count', fontsize=28, color='brown')
                ax4.tick_params(axis='y', colors='brown')
                ax4.spines['right'].set_color('brown')
                ax4.set_ylim((8700,8950))
            elif 'AACT' in name:
                axs = ax
                color='red'
                lns2 = axs.plot(group['timestamp'], group['value_numeric'],
                         label=':'.join(name.split(':')[3:]),
                         color = color, linewidth=4, alpha=0.5)
                axs.set_ylim(0,0.8)
                axs.set_ylabel('Amplitude [MV]', fontsize=28, color='red')
                axs.tick_params(axis='y', colors='red')
                axs.spines['left'].set_color('red')
            elif 'AMEAN' in name:
                ax2 = ax.twinx()
                color='blue'
                lns3 = ax2.plot(group['timestamp'], group['value_numeric'],
                                label=':'.join(name.split(':')[3:]),
                                color = color, linewidth=4, alpha=0.5)
                ax2.spines["left"].set_position(("axes", -0.05))
                ax2.yaxis.set_label_position('left')
                ax2.yaxis.set_ticks_position('left')
                ax2.set_ylabel(r'Amplitude [$\sqrt{W}$]', fontsize=28, color='blue')
                ax2.tick_params(axis='y', colors='blue')
                ax2.spines['left'].set_color('blue')
                ax2.set_ylim(7,12)
            elif 'DF:STD' in name:
                ax3 = ax.twinx()
                color='green'
                lns4 = ax3.plot(group['timestamp'], group['value_numeric'],
                         label=':'.join(name.split(':')[3:]),
                         color = color, linewidth=4, alpha=0.5)
                ax3.set_ylim(0,20)
                ax3.set_ylabel('Detuning [Hz]', fontsize=28, color='green')
                ax3.tick_params(axis='y', colors='green')
                ax3.spines['right'].set_color('green')

        lns = lns1 + lns2 + lns3 + lns4
        labs = [l.get_label() for l in lns]
        ax4.legend(lns, labs, fontsize=24, loc='upper right', framealpha=0.8)

        myFmt = mdates.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(myFmt)


        # Draw Vertical Lines for every "NANC ON" event found
        nanc_on_times = [datetime.datetime.strptime('08/20/2026 15:27:04.00', d_fmt)]
        start_time = '2026-08-20 15:24:33'
        end_time = '2026-08-20 15:29:30'
        for t_event in nanc_on_times:
            ax.axvline(x=t_event, color='grey', linestyle='--', linewidth=3)
            t_text_off = t_event - pd.Timedelta(seconds=37)
            ax.text(t_text_off, 0.7, '<-- NANC OFF ', color='grey', fontsize=24, fontweight='bold')
            ax.text(t_event, 0.7, '   NANC ON -->', color='grey', fontsize=24, fontweight='bold')

        if start_time and end_time:
            ax.set_xlim(pd.to_datetime(start_time), pd.to_datetime(end_time))
        ax.grid(True, alpha=0.3)

        ax.tick_params(axis='x')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Archive data plotter")
    parser.add_argument('-f', '--file', dest='datafile', required=True,
                        help='Waveform data file')
    parser.add_argument('-p', '--prefix', dest='prefix', required=True,
                        help='PV cavity prefix (e.g. VTS:L1B:H110:)')
    parser.add_argument("-s", "--split_index", type=int, default=None,
                        help="Index of the row to split the data")
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
    #ca_file.plot_wf('FWD')

    #ca_file.plot_rfs_res_detuning()
    #ca_file.split_plot_rfs_det(split_index=args.split_index)
    #ca_file.plot_overview()

    f1 = '../../../VTS/ca_data/ca_data/SELAP20260820_162137.txt'
    f2 = '../../../VTS/ca_data/ca_data/SELAP20260820_162646.txt'
    f3 = '../../../VTS/ca_data/ca_data/SELAP20260820_163001.txt'
    f4 = '../../../VTS/ca_data/ca_data/SELAP20260820_163549.txt'


    ca_f1 = CamonitorFile(f1, args.prefix)
    ca_f2 = CamonitorFile(f2, args.prefix)
    ca_f3 = CamonitorFile(f3, args.prefix)
    ca_f4 = CamonitorFile(f4, args.prefix)
    ca_f1.load_data()
    ca_f1.time_axis()
    ca_f2.load_data()
    ca_f2.time_axis()
    ca_f3.load_data()
    ca_f3.time_axis()
    ca_f4.load_data()
    ca_f4.time_axis()

    i_wf_f1 = ca_f1.df.loc[f"FWD:IWF", 'Value']
    q_wf_f1 = ca_f1.df.loc[f"FWD:QWF", 'Value']
    wf_f1 = np.array(i_wf_f1) + 1j * np.array(q_wf_f1)

    i_wf_f2 = ca_f2.df.loc[f"FWD:IWF", 'Value']
    q_wf_f2 = ca_f2.df.loc[f"FWD:QWF", 'Value']
    wf_f2 = np.array(i_wf_f2) + 1j * np.array(q_wf_f2)

    i_wf_f3 = ca_f3.df.loc[f"FWD:IWF", 'Value']
    q_wf_f3 = ca_f3.df.loc[f"FWD:QWF", 'Value']
    wf_f3 = np.array(i_wf_f3) + 1j * np.array(q_wf_f3)

    i_wf_f4 = ca_f4.df.loc[f"FWD:IWF", 'Value']
    q_wf_f4 = ca_f4.df.loc[f"FWD:QWF", 'Value']
    wf_f4 = np.array(i_wf_f4) + 1j * np.array(q_wf_f4)

    fig, ax = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
    print(len(wf_f4))
    f1_s = 196000
    f1_e = 199000
    f2_s = 191750
    f2_e = 194750
    f3_s = 86100
    f3_e = 89100
    f4_s = 159100
    f4_e = 162100
    
    ax[0].plot(ca_f1.taxis_rf[:f1_e-f1_s], np.abs(wf_f1[f1_s:f1_e]), label = '-2e-4')
    ax[0].plot(ca_f2.taxis_rf[:f2_e-f2_s], np.abs(wf_f2[f2_s:f2_e]), label = '-3e-4')
    ax[0].plot(ca_f3.taxis_rf[:f3_e-f3_s], np.abs(wf_f3[f3_s:f3_e]), label = '-2e-3')
    ax[0].plot(ca_f4.taxis_rf[:f4_e-f4_s], np.abs(wf_f4[f4_s:f4_e]), label = '-1e-3')

    ax[1].plot(ca_f1.taxis_rf[:f1_e-f1_s], np.degrees(np.angle(wf_f1[f1_s:f1_e])))
    ax[1].plot(ca_f2.taxis_rf[:f2_e-f2_s], np.degrees(np.angle(wf_f2[f2_s:f2_e])))
    ax[1].plot(ca_f3.taxis_rf[:f3_e-f3_s], np.degrees(np.angle(wf_f3[f3_s:f3_e])))
    ax[1].plot(ca_f4.taxis_rf[:f4_e-f4_s], np.degrees(np.angle(wf_f4[f4_s:f4_e])))

    ax[2].plot(ca_f1.taxis_rf[:f1_e-f1_s], i_wf_f1[f1_s:f1_e])
    ax[2].plot(ca_f2.taxis_rf[:f2_e-f2_s], i_wf_f2[f2_s:f2_e])
    ax[2].plot(ca_f3.taxis_rf[:f3_e-f3_s], i_wf_f3[f3_s:f3_e])
    ax[2].plot(ca_f4.taxis_rf[:f4_e-f4_s], i_wf_f4[f4_s:f4_e])

    ax[3].plot(ca_f1.taxis_rf[:f1_e-f1_s], q_wf_f1[f1_s:f1_e])
    ax[3].plot(ca_f2.taxis_rf[:f2_e-f2_s], q_wf_f2[f2_s:f2_e])
    ax[3].plot(ca_f3.taxis_rf[:f3_e-f3_s], q_wf_f3[f3_s:f3_e])
    ax[3].plot(ca_f4.taxis_rf[:f4_e-f4_s], q_wf_f4[f4_s:f4_e])

    #ax[0].set_xlim(0, ca_f1.taxis_rf[-1])
    ax[0].set_ylabel('Amplitude [MV]')
    ax[1].set_ylabel('Phase [rad]')
    ax[2].set_ylabel('I [MV]')
    ax[3].set_ylabel('Q [MV]')
    ax[3].set_xlabel('Time [s]')
    ax[0].legend()
    plt.show()