from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
import numpy as np


def read_metadata(data_f):
    metadata = {}
    cav_num = 0
    last_line = False
    with open(data_f, 'r') as f:
        for line in f:
            if 'First buffer' in line:
                break
            if not line.startswith('#'):
                last_line = True
            else:
                clean_line = line.lstrip('#').strip()

                if last_line is False:
                    if clean_line.startswith('##'):
                        cav_num += 1
                        metadata[f'CAV{cav_num}'] = {}
                        cav_number = clean_line.lstrip('##').split(' ')[2]
                        metadata[f'CAV{cav_num}']['cav_number'] = cav_number

                    if ' : ' in clean_line:
                        key, value = clean_line.split(':', 1)
                        val = value.strip()
                        try:
                            if '.' in val:
                                metadata[f'CAV{cav_num}'][key.strip()] = float(val)
                            else:
                                metadata[f'CAV{cav_num}'][key.strip()] = int(val)
                        except ValueError:
                            metadata[f'CAV{cav_num}'][key.strip()] = val
                else:
                    pv_list = clean_line.split()
                    columns = [pv.split(':')[-2] for pv in pv_list]
                    metadata['columns'] = columns
    return metadata


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


def peak_finder(data):
    indices, properties = find_peaks(data, height=0.0001)
    peak_heights = properties['peak_heights']
    top_10_idx_in_peaks = np.argsort(peak_heights)[-10:][::-1]

    actual_indices = indices[top_10_idx_in_peaks]
    actual_heights = peak_heights[top_10_idx_in_peaks]

    return actual_heights, actual_indices


def plot_detuning(time, data, label, dts, max_freq=250, req=None):
    import matplotlib.pyplot as plt
    from scipy.fft import fftfreq
    from scipy import signal

    plt.figure(1)
    plt.xlabel('Time [s]')
    plt.ylabel('Detuning [Hz]')
    if req:
        plt.axhline(y=-10, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=10, color='r', linestyle='--', alpha=0.3)
    #plt.xlim(time[0], time[-1])

    plt.figure(2)
    plt.xlabel('Frequency [Hz]')
    plt.xlim(0, max_freq)

    plt.figure(3)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Detuning PSD [Hz]')
    plt.xlim(0, max_freq)

    plt.figure(4)
    plt.xlabel('Detuning [Hz]')
    plt.ylabel('Counts')
    if req:
        plt.axvline(x=-10, color='r', linestyle='--', alpha=0.3)
        plt.axvline(x=10, color='r', linestyle='--', alpha=0.3)

    plt.figure(5)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Detuning STD [Hz]')
    plt.xlim(0, max_freq)
    plt.ylim((0, 15))

    nc = len(data)
    fig, axes = plt.subplots(1, nc, figsize=(16, 5), sharey=True)
    plt.subplots_adjust(wspace=0)

    for i, sublist in enumerate(data):
        sublist = np.array(sublist)
        N = len(sublist)
        plt.figure(1)
        plt.plot(time[i], sublist, label=label[i])

        # FFT
        fft_raw = np.fft.fft(sublist)/len(sublist)
        fft = fft_raw*dts[i]*N
        xf = fftfreq(N, dts[i])[:N//2]
        plt.figure(2)
        plt.plot(xf, 2.0/N * np.abs(fft[0:N//2]), label=label[i])

        # Power Spectral Density
        freq, psd = signal.periodogram(sublist, 1/dts[i])
        plt.figure(3)
        plt.semilogy(freq[1:], psd[1:], label=label[i])
        plt.legend(loc='upper right')
        plt.tight_layout()

        plt.figure(4)
        plt.hist(sublist, bins=140,  histtype='step', log='True', label=label[i])

        # Cumulative detuning STD
        fft_raw[0] = 0
        c_d = np.sqrt(np.cumsum(abs(fft_raw**2)))*np.sqrt(2)
        plt.figure(5)
        plt.plot(fftfreq(N, dts[i])[:N//2], c_d[:N//2], label=label[i])

        plt.figure(fig)
        f, t, Sxx = signal.spectrogram(sublist, 1/dts[i],
                                       nperseg=1000, noverlap=750)
        axes[i].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud',
                           cmap='viridis', vmin=-100, vmax=0)
        axes[i].set_title(label=label[i], size=16)
        axes[i].set_xlabel('Time [sec]')
        if i == 0:
            axes[i].set_ylabel('Frequency [Hz]')
        axes[i].set_ylim([0, 100])

    for i in range(5):
        plt.figure(i+1)
        plt.legend(loc='upper right')
        plt.tight_layout()
    plt.show()

    
    
    
