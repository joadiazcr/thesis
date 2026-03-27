import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fftfreq
from scipy import signal
from utils import read_metadata


#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')


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


def peak_finder(data):
    indices, properties = find_peaks(data, height=0.0001)
    peak_heights = properties['peak_heights']
    top_10_idx_in_peaks = np.argsort(peak_heights)[-10:][::-1]

    actual_indices = indices[top_10_idx_in_peaks]
    actual_heights = peak_heights[top_10_idx_in_peaks]

    return actual_heights, actual_indices


def mp_plot(data_f, tt):
    metadata = read_metadata(data_f)
    wsp = metadata['CAV1']['wave_samp_per']
    data_all = np.loadtxt(data_f)
    N, nch = data_all.shape
    print(f'Data shape:\t{N} X {nch}')
    print('Columns:\t' + ' | '.join(metadata['columns']))
    print(f'wsp :\t\t{wsp}\n')
    # Assume 2 kHz samp rate
    dt = wsp/2e3
    print(f'Samplig freq :\t{1/dt/1e3} kHz')
    t_base = np.arange(0, dt*N, dt)
    max_freq = 250

    data = data_all[:, 1]
    # FFT and peaks
    fft_raw = np.fft.fft(data)/len(data)
    fft = fft_raw*dt*N
    xf = fftfreq(N, dt)[:N//2]
    fft_peaks, fft_peaks_idx = peak_finder(2.0/N * np.abs(fft[0:N//2]))

    # Power Spectral Density
    freq, psd = signal.periodogram(data, 1/dt)

    # Cumulative detuning STD
    fft_raw[0] = 0
    c_d = np.sqrt(np.cumsum(abs(fft_raw**2)))*np.sqrt(2)

    up_peak = np.amax(data)
    lw_peak = np.amax(-data)
    peak = max(up_peak, lw_peak)

    print(f'Time length:\t{N*dt:.2f} seconds')
    print(f'Peak detuning:\t{peak:.2f} Hz')
    print(f'Detuning STD:\t{np.std(data):.2f} Hz\n')
    print(f'FFT peaks: {fft_peaks}')

    plt.figure(1)
    if tt != '':
        title = 'Cavity Detuning\n' + tt
        plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Detuning [Hz]')
    plt.plot(t_base, data)
    plt.axhline(y=-10, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=10, color='r', linestyle='--', alpha=0.3)
    plt.xlim(t_base[0], t_base[-1])
    plt.tight_layout()

    plt.figure(2)
    if tt != '':
        title = 'Cavity Detuning Spectrum\n' + tt
        plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.plot(xf, 2.0/N * np.abs(fft[0:N//2]))
    plt.scatter(xf[fft_peaks_idx], fft_peaks, s=100, marker='*', color='orange')
    plt.xlim(0, max_freq)
    plt.tight_layout()

    plt.figure(3)
    if tt != '':
        title = 'Cavity Detuning PSD\n' + tt
        plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Detuning PSD [Hz]')
    plt.semilogy(freq[1:], psd[1:])
    plt.xlim(0, max_freq)
    plt.tight_layout()

    plt.figure(4)
    if tt != '':
        title = 'Detuning Histogram\n' + tt
        plt.title(title)
    plt.xlabel('Detuning [Hz]')
    plt.ylabel('Counts')
    plt.hist(data, bins=140,  histtype='step', log='True')
    plt.axvline(x=-10, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=10, color='r', linestyle='--', alpha=0.3)
    plt.tight_layout()

    plt.figure(5)
    if tt != '':
        title = 'Cumulative Detuning STD\n' + tt
        plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Detuning STD [Hz]')
    plt.plot(fftfreq(N, dt)[:N//2], c_d[:N//2])
    plt.xlim(0, max_freq)
    plt.ylim(bottom=0)
    plt.tight_layout()

    plt.figure(6)
    f, t, Sxx = signal.spectrogram(data, 1/dt, nperseg=1000, noverlap=750)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis', vmin=-100, vmax=0)
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.ylim([0,100])

    plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Microphonics plotter")
    parser.add_argument('-f', '--file', dest='datafile', required=True,
                        help='Waveform data file')
    parser.add_argument('-t', '--title', dest='tt', help='Plot Title',
                        default='')

    args = parser.parse_args()

    if args.tt == '':
        plt.rc('font', size=18)
        plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
        plt.rc('legend', fontsize=14)    # legend fontsize
        plt.rc('figure', titlesize=18)  # fontsize of the figure title

    print(f'Filename :\t{args.datafile}')
    mp_plot(args.datafile, args.tt)
