import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.dates as mdates


#plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def archive_plot():
    file_prefix = '~/Documents/NANC/shift_20251106/shift_20251106_'
    cm19_cav3 = True
    if cm19_cav3 == True:
        file_app = 'cm19_cav3_'
        pvprefix = 'ACCL:L3B:1930:'
        ylim_clip_limit = (0,20)
        nanc_switch = dt.datetime(2025, 11, 6, 14, 4, 58)
        txt_nanc_off = dt.datetime(2025, 11, 6, 14, 4, 0)
        txt_nanc_on = dt.datetime(2025, 11, 6, 14, 5, 20)
        start = np.datetime64("2025-11-06T14:00:00")
        end = np.datetime64("2025-11-06T14:10:00")
    else:
        file_app = ''
        pvprefix = 'ACCL:L3B:1920:'
        ylim_clip_limit = (0,800)
        nanc_switch = dt.datetime(2025, 11, 6, 14, 55, 5)
        txt_nanc_off = dt.datetime(2025, 11, 6, 14, 54, 0)
        txt_nanc_on = dt.datetime(2025, 11, 6, 14, 55, 20)
        start = np.datetime64("2025-11-06T14:47:00")
        end = np.datetime64("2025-11-06T15:01:00")
    
    df_fwd_amean = pd.read_csv(file_prefix + file_app + 'fwd_amean',
                     sep='\t', comment='#', index_col=False,  parse_dates=['Timestamp'])
    df_ades = pd.read_csv(file_prefix + file_app + 'ades',
                     sep='\t', comment='#', index_col=False,  parse_dates=['Timestamp'])
    df_cav_amean = pd.read_csv(file_prefix + file_app + 'aactmean',
                     sep='\t', comment='#', index_col=False,  parse_dates=['Timestamp'])
    df_df_std = pd.read_csv(file_prefix + file_app + 'df_std',
                     sep='\t', comment='#', index_col=False,  parse_dates=['Timestamp'])
    df_clip_cnt = pd.read_csv(file_prefix + file_app + 'fb_sum_cntt',
                     sep='\t', comment='#', index_col=False,  parse_dates=['Timestamp'])

    t_fwd_amean = df_fwd_amean['Timestamp'].to_numpy()
    y_fwd_amean = df_fwd_amean[pvprefix + 'FWD:AMEAN'].to_numpy()
    t_ades = df_ades['Timestamp'].to_numpy()
    y_ades = df_ades[pvprefix + 'ADES'].to_numpy()
    t_cav_amean = df_cav_amean['Timestamp'].to_numpy()
    y_cav_amean = df_cav_amean[pvprefix + 'AACTMEAN'].to_numpy()
    t_df_std = df_df_std['Timestamp'].to_numpy()
    y_df_std = df_df_std[pvprefix + 'DF:STD'].to_numpy()
    t_clip_cnt = df_clip_cnt['Timestamp'].to_numpy()
    y_clip_cnt = df_clip_cnt[pvprefix + 'FB_SUM_CNTT'].to_numpy()

    fig, ax = plt.subplots()
    lns1 = ax.step(t_ades, y_ades, '--', label='ADES', color='black', linewidth=2, where='post')
    lns2 = ax.plot(t_cav_amean, y_cav_amean, label='AACTMEAN', color='red', linewidth=4, alpha=0.5)
    ax.set_ylim(0,16)
    ax.set_xlabel('Time', fontsize=24)
    ax.set_ylabel('Amplitude [MV]', fontsize=24, color='red')
    ax.tick_params(axis='y', colors='red')
    ax.spines['left'].set_color('red')
    
    ax2 = ax.twinx()
    lns3 = ax2.plot(t_fwd_amean, y_fwd_amean, color='blue', label="FWD:AMEAN", linewidth=4, alpha=0.5)
    ax2.spines["left"].set_position(("axes", -0.05))
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.set_ticks_position('left')
    ax2.set_ylabel(r'Amplitude [$\sqrt{W}$]', fontsize=24, color='blue')
    ax2.tick_params(axis='y', colors='blue')
    ax2.spines['left'].set_color('blue')
    ax2.set_ylim(0,70)

    ax3 = ax.twinx()
    lns4 = ax3.plot(t_df_std, y_df_std, label="DF:STD", color='green', linewidth=4, alpha=0.5)
    ax3.set_ylim(0,50)
    ax3.set_ylabel('Detuning [Hz]', fontsize=24, color='green')
    ax3.tick_params(axis='y', colors='green')
    ax3.spines['right'].set_color('green')

    ax4 = ax.twinx()
    lns5 = ax4.step(t_clip_cnt, y_clip_cnt, label="FB_SUM_CNTT", color='brown', linewidth=4, alpha=0.8)
    ax4.spines["right"].set_position(("axes", 1.05))
    ax4.set_ylabel('Clip limit count', fontsize=24, color='brown')
    ax4.tick_params(axis='y', colors='brown')
    ax4.spines['right'].set_color('brown')
    ax4.set_ylim(ylim_clip_limit)

    ax.set_xlim(start, end)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.axvline(x=nanc_switch, color='grey', linestyle='--', linewidth=3)

    lns = lns1 + lns2 + lns3 + lns4 + lns5 
    labs = [l.get_label() for l in lns]
    ax4.legend(lns, labs, fontsize=18, loc='upper right', framealpha=1.0)
    ax.text(txt_nanc_off, 15.5, '<-- NANC OFF', color='black', fontsize=16)
    ax.text(txt_nanc_on, 15.5, 'NANC ON -->', color='black', fontsize=16)

    plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Archive data plotter")
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

    archive_plot()