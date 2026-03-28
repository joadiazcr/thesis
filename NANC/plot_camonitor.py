from dataclasses import dataclass, field
import os
import pandas as pd
import matplotlib.pyplot as plt


d_fmt = "%m/%d/%Y %H:%M:%S.%f"
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')


@dataclass
class CamonitorFile():
    filename: str
    df: pd.DataFrame = field(default=None, repr=False)

    def load_data(self):
        """Parses the file into a Pandas DataFrame."""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Could not find {self.filename}")

        variable_names = ['variable', 'date', 'time', 'val1', 'val2']
        self.df = pd.read_csv(self.filename,
                              sep=r'\s+',
                              names=variable_names,
                              usecols=[0, 1, 2, 3, 4],
                              on_bad_lines='warn',
                              parse_dates=[['date', 'time']])

        self.df['value_raw'] = self.df['val1'].astype(str) +\
            " " +\
            self.df['val2'].fillna('').astype(str)
        self.df['value_raw'] = self.df['value_raw'].str.strip()

        status_map = {
            'Enabled': 1,
            'Not enabled': 0
        }
        self.df['value_raw'] = self.df['value_raw'].replace(status_map)

        self.df['value_numeric'] = pd.to_numeric(self.df['value_raw'],
                                                 errors='coerce')

        self.df = self.df.rename(columns={'date_time': 'timestamp'})

        data_arrays = {}
        for var_name, group in self.df.groupby('variable'):
            data_arrays[var_name] = {
                'x': group['timestamp'].values,
                'y': group['value_numeric'].values
            }

    def plot_data(self, target_pvs=None, start_time=None, end_time=None):
        """
        Plots the loaded data.
        target_pvs: List of strings (e.g., ['AMAX', 'AMEAN']). If None, plots all.
        start_time / end_time: String format 'YYYY-MM-DD HH:MM:SS' or datetime object
        """
        plot_df = self.df
        # Identify "NANC ON" Events (Transitions 0 -> 1)
        # We look specifically for the ENABLESTAT variable
        en_data = plot_df[plot_df['variable'].str.contains('ENABLESTAT')].sort_values('timestamp')
        # .diff() == 1 means the value went from 0 to 1
        nanc_on_times = en_data[en_data['value_numeric'].diff() == 1]['timestamp']

        # Filter for specific PVs if requested
        if target_pvs:
            mask = plot_df['variable'].str.contains('|'.join(target_pvs))
            plot_df = plot_df[mask]

        # Get unique variables to plot
        groups = plot_df.groupby('variable')

        fig, ax = plt.subplots(figsize=(10, 6))
        color = 'black'
        for name, group in groups:
            if 'CNTU' in name:
                ax4 = ax.twinx()
                lns5 = ax4.step(group['timestamp'], group['value_numeric'],
                                label=name, color='brown', linewidth=4, alpha=0.8)
            elif 'AACT' in name:
                axs = ax
                color='red'
                axs.plot(group['timestamp'], group['value_numeric'], 
                         color = color,
                         label=name)
            else:
                axs = ax
                axs.plot(group['timestamp'], group['value_numeric'], 
                         color = color,
                         label=name)
        ax.legend()

        # Draw Vertical Lines for every "NANC ON" event found
        for t_event in nanc_on_times:
            ax.axvline(x=t_event, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

            ax.text(t_event, 0.9, 'NANC ON',
                     color='red', fontsize=10, fontweight='bold',
                     verticalalignment='top')

        if start_time and end_time:
            ax.set_xlim(pd.to_datetime(start_time), pd.to_datetime(end_time))
        ax.grid(True, alpha=0.3)
        ax.set_title("Combined PV Trends")

        ax.set_xlabel("Timestamp")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Plot data acquired with camonitor")
    parser.add_argument('-f', '--file', dest='datafile', required=True,
                        help='StripTool data file')
    parser.add_argument("-ch", "--ch_list", dest="ch_list", nargs="+", action="store",
                        default=["DAC", "DF"],
                        help="Override channel acq list. ALL to enable all")

    args = parser.parse_args()

    ca_file = CamonitorFile(args.datafile)
    ca_file.load_data()
    ca_file.plot_data(target_pvs=['AACTMEAN', 'CNTU', 'AMAX', 'AMEAN', 'RANGE', 'STD'])#,
                      #start_time='2026-03-19 14:53:00',
                      #end_time='2026-03-19 14:56:00')
