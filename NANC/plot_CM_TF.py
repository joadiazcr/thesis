import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils import read_metadata
from plot_TF import ResData


plt.rc('font', family='serif')
plt.rc('font', size=22)
plt.rc('axes', labelsize=26)
plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)
plt.rc('legend', fontsize=22)
plt.rc('figure', titlesize=22)


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Transfer Function plotter")
    parser.add_argument('-p', '--path', dest='datafile', required=True,
                        help='Path to resonance control data files')
    parser.add_argument('-c', '--count', dest='count', required=True, type=int,
                        help='Number of chirps in each res file')
    parser.add_argument('-wsp', '--wave_samp_per', dest='wsp', default=1,
                        type=int, help='Waveform decimation factor')
    args = parser.parse_args()

    dir_path = Path(args.datafile)
    files = [f for f in dir_path.iterdir() if f.is_file()]
    files.sort(key=lambda f: f.name)

    nominal_low_freq = 1
    nominal_high_freq = 80
    amp_max = 0
    fig1 = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig2 = plt.figure(figsize=(19.2, 10.8), dpi=100)
    for file in files:
        metadata = read_metadata(file)
        cav_num = metadata['CAV1']['cav_number']
        
        parts = file.name.split('_')
        low_freq = int(parts[2].replace('Hz', ''))
        high_freq = int(parts[3].replace('Hz', ''))

        if low_freq == nominal_low_freq and high_freq == nominal_high_freq:
            print(file.name)
            res_data = ResData(file, args.count, args.wsp)
            amp_max_temp = np.max(res_data.tf_mag[nominal_low_freq:nominal_high_freq*3])
            if amp_max_temp > amp_max:
                amp_max = amp_max_temp
                print(amp_max)
            plt.figure(fig1.number)
            plt.plot(res_data.p_dac.xf, res_data.tf_mag, linewidth=2, label=f'cav{cav_num}')
            plt.figure(fig2.number)
            plt.plot(res_data.p_dac.xf, res_data.tf_phs, linewidth=2, label=f'cav{cav_num}')
    
    plt.figure(fig1.number)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('A [Hz/V]')
    plt.ylim(0, 1.1 * amp_max)
    plt.xlim(nominal_low_freq, nominal_high_freq)
    plt.legend()

    plt.figure(fig2.number)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [rad]')
    plt.ylim(-2*np.pi, 2*np.pi)
    plt.xlim(nominal_low_freq, nominal_high_freq)
    plt.legend()
    plt.show()