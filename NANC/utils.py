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