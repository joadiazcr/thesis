import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq
from scipy import signal
from scipy.signal import butter, filtfilt

plt.rc('font', family='serif', size=18)
plt.rc('mathtext', fontset='cm')
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=14)
plt.rc('figure', titlesize=18)


def butter_highpass_filter(data, cutoff, fs, order=5):
    """
    data: Your input signal
    cutoff: The frequency below which signals are blocked (Hz)
    fs: The sampling rate of your data (Hz)
    order: The 'steepness' of the filter
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq

    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    # Apply the filter
    # 'filtfilt' is better than 'lfilter' because it applies the filter
    # twice (forward and backward) to eliminate phase shift/delay.
    y = filtfilt(b, a, data)
    return y


def psd_calc(y, dt):
    npt = len(y)
    window = np.hanning(npt)*2
    ss = np.fft.fft((y - np.mean(y)) * window)
    ss = np.array(ss[0:npt//2]) / sum(window) * np.sqrt(4.0/3.0)
    print(sum(np.abs(ss)**2))
    df = 1.0/npt/dt
    freq = np.arange(npt/2) * df
    psd = abs(ss)**2/df

    # correct for the sinc^2 filter that's used
    # for anti-aliasing before decimation
    st = np.sinc(freq*dt)**4 + np.sinc(1-freq*dt)**4
    psd = psd / st

    # throw in the software-defined BBF 1 Hz high-pass, first-order
    f_highpass = 1.0  # Hz
    norm_f2 = (freq/f_highpass)**2
    bbf = norm_f2 / (norm_f2+1)
    psd_fake = psd * bbf
    psd_fake[1:3] = 0

    psd_integral = np.cumsum(psd_fake*df)
    return freq, psd, psd_integral


def mp_plot(data_f, wsp, tt, num_cavities):
    data_all = np.loadtxt(data_f)
    N, nch = data_all.shape
    print('data shape: %s X %s' % (N, nch))
    # Assume 2 kHz samp rate
    dt = wsp/2e3
    print(f'Samplig freq :\t{1/dt/1e3} kHz')
    print(f'Time length:\t{N*dt:.2f} seconds\n')
    t_base = np.arange(0, dt*N, dt)
    max_freq = 250

    n_columns_per_file = 4
    num_cavities = 4

    fig1, ax1 = plt.subplots(1, 4, figsize=(20, 10),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    fig2, ax2 = plt.subplots(1, 4, figsize=(20, 10),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    fig3, ax3 = plt.subplots(1, 4, figsize=(20, 10),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    fig4, ax4 = plt.subplots(1, 4, figsize=(20, 10),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    fig5, ax5 = plt.subplots(1, 4, figsize=(20, 10),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    fig6, ax6 = plt.subplots(1, 8, figsize=(20, 10),
                             sharex=True, sharey=True,
                             constrained_layout=True)

    for i in range(num_cavities):
        print(f'Cavity {i+1}')
        offset = 0 if int(i/4) == 0 else 32
        corr = 0 if int(i/4) == 0 else 4
        data = data_all[:, int(((i-corr)*n_columns_per_file)+1+offset)]
        data_on = data_all[:, int(((i-corr)*n_columns_per_file)+1+16+offset)]

        # FFT
        fft_raw = np.fft.fft(data)/len(data)
        fft = fft_raw*dt*N
        xf = fftfreq(N, dt)[:N//2]

        fft_raw_on = np.fft.fft(data_on)/len(data_on)
        fft_on = fft_raw_on*dt*N

        # Power Spectral Density
        freq, psd = signal.periodogram(data_on, 1/dt)
        freq_on, psd_on = signal.periodogram(data_on, 1/dt)

        # Cumulative detuning STD
        fft_raw[0] = 0
        c_d = np.sqrt(np.cumsum(abs(fft_raw**2)))*np.sqrt(2)

        fft_raw_on[0] = 0
        c_d_on = np.sqrt(np.cumsum(abs(fft_raw_on**2)))*np.sqrt(2)

        up_peak = np.amax(data)
        lw_peak = np.amax(-data)
        peak = max(up_peak, lw_peak)

        up_peak_on = np.amax(data_on)
        lw_peak_on = np.amax(-data_on)
        peak_on = max(up_peak_on, lw_peak_on)

        print(f'Peak detuning:\t{peak:.2f} Hz')
        print(f'Detuning STD:\t{np.std(data):.2f} Hz\n')
        print(f'Peak detuning NANC ON:\t{peak_on:.2f} Hz')
        print(f'Detuning STD NANC ON:\t{np.std(data_on):.2f} Hz\n')

        r = int(i/4)
        c = i % 4

        # Plot detuning
        ax1[c].plot(t_base, data, label='NANC OFF')
        ax1[c].plot(t_base, data_on, label='NANC ON CAV1')
        ax1[c].set_title(f'Cavity {i+1}', fontsize=16)
        if r == 1:
            ax1[c].set_xlabel('Time [s]')
        if c == 0:
            ax1[c].set_ylabel('Detuning [Hz]')
        ax1[c].axhline(y=-10, color='r', linestyle='--', alpha=0.3)
        ax1[c].axhline(y=10, color='r', linestyle='--', alpha=0.3)
        ax1[c].set_xlim(t_base[0], t_base[-1])

        # Plot FFT
        ax2[c].plot(xf, 2.0/N * np.abs(fft[0:N//2]))
        ax2[c].plot(xf, 2.0/N * np.abs(fft_on[0:N//2]))
        ax2[c].set_title(f'Cavity {i+1}', fontsize=16)
        if r == 1:
            ax2[c].set_xlabel('Frequency [Hz]')
        ax2[c].set_xlim(0, max_freq)

        # Plot PSD
        ax3[c].semilogy(freq[1:], psd[1:], label=f'cav{i+1}')
        ax3[c].semilogy(freq[1:], psd_on[1:], label=f'cav{i+1}')
        ax3[c].set_title(f'Cavity {i+1}', fontsize=16)
        if r == 1:
            ax2[c].set_xlabel('Frequency [Hz]')
        if c == 0:
            ax3[c].set_ylabel('Detuning PSD [Hz]')
        ax3[c].set_xlim(0, max_freq)

        # Plot histogram
        ax4[c].hist(data, bins=140,  histtype='step', log='True',
                    label=f'cav{i+1}')
        ax4[c].hist(data_on, bins=140,  histtype='step', log='True',
                    label=f'cav{i+1}')
        ax4[c].set_title(f'Cavity {i+1}', fontsize=16)
        if r == 1:
            ax4[c].set_xlabel('Detuning [Hz]')
        if c == 0:
            ax4[c].set_ylabel('Counts')
        ax4[c].axvline(x=-10, color='r', linestyle='--', alpha=0.3)
        ax4[c].axvline(x=10, color='r', linestyle='--', alpha=0.3)

        # Plot STD
        ax5[c].plot(fftfreq(N, dt)[:N//2], c_d[:N//2], label=f'cav{i+1}')
        ax5[c].plot(fftfreq(N, dt)[:N//2], c_d_on[:N//2], label=f'cav{i+1}')
        ax5[c].set_title(f'Cavity {i+1}', fontsize=16)
        if r == 1:
            ax5[c].set_xlabel('Frequency [Hz]')
        if c == 0:
            ax5[c].set_ylabel('Detuning STD [Hz]')
        ax5[c].set_xlim(0, max_freq)

        # Plot Spectrogram
        f, t, Sxx = signal.spectrogram(data, 1/dt, nperseg=1024, noverlap=768)
        f_on, t_on, Sxx_on = signal.spectrogram(data_on, 1/dt,
                                                nperseg=1024, noverlap=768)
        ax6[(c*2)].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud',
                              cmap='viridis', vmin=-100, vmax=0)
        ax6[(c*2)+1].pcolormesh(t_on, f_on, 10 * np.log10(Sxx_on),
                                shading='gouraud',
                                cmap='viridis', vmin=-100, vmax=0)
        ax6[(c*2)].set_title(f'Cavity {i+1}', fontsize=16)
        ax6[(c*2)+1].set_title(f'Cavity {i+1} - NANC ON CAV1', fontsize=16)
        if r == 1:
            ax6[(c*2)].set_xlabel('Time [s]')
            ax6[(c*2)+1].set_xlabel('Time [s]')
        if c == 0:
            ax6[c].set_ylabel('Frequency [Hz]')
        ax6[c].set_ylim([0, 100])

    handles, labels = ax1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper center', fontsize=16, ncol=2)
    fig1.get_layout_engine().set(h_pad=0.2, w_pad=0.2, rect=(0, 0, 1, 0.95))
    fig2.legend(handles, labels, loc='upper center', fontsize=16, ncol=2)
    fig2.get_layout_engine().set(h_pad=0.2, w_pad=0.2, rect=(0, 0, 1, 0.95))
    fig3.legend(handles, labels, loc='upper center', fontsize=16, ncol=2)
    fig3.get_layout_engine().set(h_pad=0.2, w_pad=0.2, rect=(0, 0, 1, 0.95))
    fig4.legend(handles, labels, loc='upper center', fontsize=16, ncol=2)
    fig4.get_layout_engine().set(h_pad=0.2, w_pad=0.2, rect=(0, 0, 1, 0.95))
    fig5.legend(handles, labels, loc='upper center', fontsize=16, ncol=2)
    fig5.get_layout_engine().set(h_pad=0.2, w_pad=0.2, rect=(0, 0, 1, 0.95))

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
    parser.add_argument('-t', '--title', dest='tt', help='Plot Title',
                        default='')

    args = parser.parse_args()

    mp_plot(args.datafile, args.wsp, args.tt, args.num_cavs)
