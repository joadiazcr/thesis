import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq
from scipy import signal

plt.rc('font', family='serif', size=18)
plt.rc('mathtext', fontset='cm')
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=14)
plt.rc('figure', titlesize=18)


def mp_plot(data_f, wsp, tt, num_cavities, col_per_cav, label_list):
    dt = wsp/2e3  # 2 kHz default sampling rate
    data_all = np.loadtxt(data_f)
    N, nch = data_all.shape
    iterations = nch // col_per_cav // num_cavities
    print(f'Data shape: {N} X {nch}')
    print(f'Iterations: {iterations}')
    print(f'Samplig freq :\t{1/dt/1e3} kHz')
    print(f'Time length:\t{N*dt:.2f} seconds\n')

    t_base = np.arange(0, dt*N, dt)
    max_freq = 250

    n_plot_cols = int(num_cavities if num_cavities < 5 else num_cavities / 2)

    config = {
        "detuning": n_plot_cols,
        "fft": n_plot_cols,
        "psd": n_plot_cols,
        "histogram": n_plot_cols,
        "std": n_plot_cols,
        "spectrogram": n_plot_cols * 2
    }
    plots = {}

    # Create figures
    for name, num_cols in config.items():
        fig, ax = plt.subplots(1, num_cols, figsize=(20, 5),
                               sharex=True, sharey=True,
                               layout="constrained")
        plots[name] = {"fig": fig, "ax": ax}

    for i in range(num_cavities):
        print(f'Cavity {i+1}')
        for j in range(iterations):
            index =  int((i*col_per_cav)+1+(j*num_cavities*col_per_cav))
            data = data_all[:, index]

            # FFT
            fft_raw = np.fft.fft(data)/len(data)
            fft = fft_raw*dt*N
            xf = fftfreq(N, dt)[:N//2]

            # Power Spectral Density
            freq, psd = signal.periodogram(data, 1/dt)

            # Cumulative detuning STD
            fft_raw[0] = 0
            c_d = np.sqrt(np.cumsum(abs(fft_raw**2)))*np.sqrt(2)

            up_peak = np.amax(data)
            lw_peak = np.amax(-data)
            peak = max(up_peak, lw_peak)

            print(f'\t{label_list[j]}')
            print(f'\t\tPeak detuning:\t{peak:.2f} Hz')
            print(f'\t\tDetuning STD:\t{np.std(data):.2f} Hz')

            r = int(i/4)
            c = i % 4

            # Plot detuning
            plots["detuning"]["ax"][c].plot(t_base, data, label=label_list[j])
            plots["detuning"]["ax"][c].set_title(f'Cavity {i+1}', fontsize=16)
            if r == 1:
                plots["detuning"]["ax"][c].set_xlabel('Time [s]')
            if c == 0:
                plots["detuning"]["ax"][c].set_ylabel('Detuning [Hz]')
            plots["detuning"]["ax"][c].axhline(y=-10, color='r', linestyle='--')
            plots["detuning"]["ax"][c].axhline(y=10, color='r', linestyle='--')
            plots["detuning"]["ax"][c].set_xlim(t_base[0], t_base[-1])

            # Plot FFT
            plots["fft"]["ax"][c].plot(xf, 2.0/N * np.abs(fft[0:N//2]))
            plots["fft"]["ax"][c].set_title(f'Cavity {i+1}', fontsize=16)
            if r == 1:
                plots["fft"]["ax"][c].set_xlabel('Frequency [Hz]')
            plots["fft"]["ax"][c].set_xlim(0, max_freq)

            # Plot PSD
            plots["psd"]["ax"][c].semilogy(freq[1:], psd[1:], label=f'cav{i+1}')
            plots["psd"]["ax"][c].set_title(f'Cavity {i+1}', fontsize=16)
            if r == 1:
                plots["psd"]["ax"][c].set_xlabel('Frequency [Hz]')
            if c == 0:
                plots["psd"]["ax"][c].set_ylabel('Detuning PSD [Hz]')
            plots["psd"]["ax"][c].set_xlim(0, max_freq)

            # Plot histogram
            plots["histogram"]["ax"][c].hist(data, bins=140,  histtype='step',
                                            log='True')
            plots["histogram"]["ax"][c].set_title(f'Cavity {i+1}', fontsize=16)
            if r == 1:
                plots["histogram"]["ax"][c].set_xlabel('Detuning [Hz]')
            if c == 0:
                plots["histogram"]["ax"][c].set_ylabel('Counts')
            plots["histogram"]["ax"][c].axvline(x=-10, color='r', linestyle='--')
            plots["histogram"]["ax"][c].axvline(x=10, color='r', linestyle='--')

            # Plot STD
            plots["std"]["ax"][c].plot(fftfreq(N, dt)[:N//2], c_d[:N//2])
            plots["std"]["ax"][c].set_title(f'Cavity {i+1}', fontsize=16)
            if r == 1:
                plots["std"]["ax"][c].set_xlabel('Frequency [Hz]')
            if c == 0:
                plots["std"]["ax"][c].set_ylabel('Detuning STD [Hz]')
            plots["std"]["ax"][c].set_xlim(0, max_freq)

            # Plot Spectrogram
            f, t, Sxx = signal.spectrogram(data, 1/dt, nperseg=1024, noverlap=768)
            plots["spectrogram"]["ax"][(c*2)].pcolormesh(t, f, 10 * np.log10(Sxx),
                                                        shading='gouraud',
                                                        cmap='viridis',
                                                        vmin=-100, vmax=0)
            plots["spectrogram"]["ax"][(c*2)].set_title(f'Cavity {i+1}',
                                                        fontsize=16)
            plots["spectrogram"]["ax"][(c*2)+1].set_title(f'Cavity {i+1} - NANC ON CAV1',
                                                        fontsize=16)
            if r == 1:
                plots["spectrogram"]["ax"][(c*2)].set_xlabel('Time [s]')
                plots["spectrogram"]["ax"][(c*2)+1].set_xlabel('Time [s]')
            if c == 0:
                plots["spectrogram"]["ax"][c].set_ylabel('Frequency [Hz]')
            plots["spectrogram"]["ax"][c].set_ylim([0, 100])

    handles, labels = plots["detuning"]["ax"][0].get_legend_handles_labels()
    for name, item in plots.items():
        f = item["fig"]
        f.legend(handles, labels, loc='upper center', fontsize=16, ncol=3)
        f.get_layout_engine().set(h_pad=0.2, w_pad=0.2, rect=(0, 0, 1, 0.95))

    plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Microphonics plotter")
    parser.add_argument('-f', '--file', dest='datafile', required=True,
                        help='Waveform data file')
    parser.add_argument('-wsp', '--wave_samp_per', dest='wsp', required=True,
                        type=int, help='Waveform decimation factor')
    parser.add_argument('-n', '--num_cavs', required=True,
                        type=int, help='Number of cavities')
    parser.add_argument('-c', '--columns', required=True,
                        type=int, help='Number of columns per cavity')
    parser.add_argument('-l', '--label_list', nargs='+', required=True,
                        help='List of labels')
    parser.add_argument('-t', '--title', dest='tt', help='Plot Title',
                        default='')

    args = parser.parse_args()

    mp_plot(args.datafile, args.wsp, args.tt, args.num_cavs,
            args.columns, args.label_list)
