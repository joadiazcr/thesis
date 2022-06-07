import matplotlib.pyplot as plt
import pandas as pd
import argparse

bunch_rate = 3e6  # Hz
bunch_T = 1/bunch_rate


def bunch2us(x):
    return x * bunch_T * 1e6


def us2bunch(x):
    return x / (bunch_T * 1e6)


def single_BPM(data_file):
    fig, axs = plt.subplots(figsize=(7, 6))
    bpm = 5
    bpm_name = BPMs[bpm]
    for v125 in range(0, len(df['V125'])):
        bpm_mean = df[bpm_name + '_data'][v125]
        bpm_mean_mean = df[bpm_name + '_mean'][v125]
        label = (str(df['V125'][v125]) + ' A')
        axs.plot(bpm_mean - bpm_mean_mean, label=label)
    axs.set_xlabel('Bunch Number', fontsize=16, fontweight='bold')
    axs.set_ylabel('Position ' + r'$\mathbf{[\mu m]}$', fontsize=16,
                   fontweight='bold')
    secax = axs.secondary_xaxis('top', functions=(bunch2us, us2bunch))
    secax.set_xlabel('Time ' + r'$\mathbf{[\mu s]}$', fontsize=16,
                     fontweight='bold')
    secax.set_xticklabels(['0', '0', '2', '4', '6',
                           '8', '10', '12', '14', '16'],
                          fontsize=14, fontweight='bold')

    legend_properties = {'weight': 'bold', 'size': 14}
    plt.legend(loc='upper right', prop=legend_properties)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.grid()
    plt.show()


def multiple_BPM(data_file, BPMs):
    num_BPMs = len(BPMs)
    num_rows = 2
    num_cols = 4

    fig, axs = plt.subplots(num_rows, num_cols, sharey=True, sharex=True)
    title = ('HEBPMS Upstream 250pC/b 50b H125@reference')
    fig.suptitle(title, fontsize=20)  # Hardcoded title
    for bpm in range(0, num_BPMs):
        row = int(bpm / num_cols)
        column = bpm % num_cols
        bpm_name = BPMs[bpm]
        title = bpm_name
        for v125 in range(0, len(df['V125'])):
            bpm_mean = df[bpm_name + '_data'][v125]
            bpm_mean_mean = df[bpm_name + '_mean'][v125]
            label = (str(df['V125'][v125]) + ' A')
            axs[row, column].plot(bpm_mean - bpm_mean_mean, label=label)
        axs[row, column].grid(True)
        title = (bpm_name)
        axs[row, column].set_title(title, fontsize=18)
        axs[row, column].tick_params(axis='both', which='major', labelsize=16)
        if int(bpm / 4) == 1:
            axs[int(bpm / 4), bpm % 4].set_xlabel('Bunch Number', fontsize=18)
        if bpm % 4 == 0:
            axs[int(bpm / 4), bpm % 4].set_ylabel('Position (um)', fontsize=18)

        plt.legend(loc='upper right', fontsize=14)

    plt.show()


if __name__ == "__main__":
    des = 'Plot BPM data'
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('-f', '--file', dest='data_file', required=True,
                        help='File with data to be plotted')
    parser.add_argument('-c', '--con', dest='connection', required=True,
                        help='HOM connection: Upstream or Downstream')
    parser.add_argument('-r', '--ref', dest='ref', required=True,
                        help='Magnet at reference: V125 or H125')
    parser.add_argument('-b', '--bc', dest='bunch_charge', required=True,
                        type=int, help='Bunch Charge')
    parser.add_argument('-n', '--nb', dest='num_bunches', required=True,
                        type=int, help='Number of bunches')

    args = parser.parse_args()

    df = pd.read_pickle(args.data_file)
    df = df[df['bunch_charge'] == args.bunch_charge]
    df = df[df['connection'] == args.connection]
    df = df[df[args.ref] == 0.0]
    df = df[df['num_bunches'] == args.num_bunches]

    df = df.reset_index(drop=True)

    BPMs = ['B418PH', 'B418PV', 'B440PH', 'B440PV',
            'B441PH', 'B441PV', 'B444PH', 'B444PV']

    single_BPM(df)
    multiple_BPM(df, BPMs)
