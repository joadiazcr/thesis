import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.fft import fft, fftfreq
from scipy import signal

def psd_calc(y, dt):
    npt = len(y)
    window = np.hanning(npt)*2
    ss = np.fft.fft((y - np.mean(y)) * window)
    ss = np.array(ss[0:npt//2]) / sum(window) * np.sqrt(4.0/3.0)
    print(sum(np.abs(ss)**2))
    df = 1.0/npt/dt
    freq = np.arange(npt/2) * df
    psd = abs(ss)**2/df

    # correct for the sinc^2 filter that's used for anti-aliasing before decimation
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


def mp_plot(data_f, wsp, conf_f, tt):

    f = open(conf_f)
    conf = json.load(f)
    exps = []
    labels = []
    for i in conf:
        exps = np.append(exps, i)
        labels = np.append(labels, conf[i])

    data = np.loadtxt(data_f)
    N, nch = data.shape
    data1 = data[:,int(exps[1])]
    label1 = labels[1]
    data2 = data[:,int(exps[0])]
    label2 = labels[0]

    # Assume 2 kHz samp rate
    dt = wsp/2e3
    fs = 1/dt
    t_base = np.arange(0, dt*N, dt)

    # FFT
    fft_raw = np.fft.fft(data1)/len(data1)
    fft_raw2 = np.fft.fft(data2)/len(data2)
    fft = fft_raw*dt*N
    fft2 = fft_raw2*dt*N
    xf = fftfreq(N, dt)[:N//2]

    # Power Spectral Density
    freq, psd = signal.periodogram(data1, 1/dt)
    freq2, psd2 = signal.periodogram(data2, 1/dt)

    # Cumulative detuning STD
    fft_raw[0]=0
    fft_raw2[0]=0
    c_d = np.sqrt(np.cumsum(abs(fft_raw**2)))*np.sqrt(2)
    c_d2 = np.sqrt(np.cumsum(abs(fft_raw2**2)))*np.sqrt(2)

    max_freq = 250
    plt.figure(1)
    title = 'Cavity Detuning\n' + tt
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Detuning [Hz]')
    plt.plot(t_base, data1, label = label1)
    plt.plot(t_base, data2, label = label2)
    plt.axhline(y=-10, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=10, color='r', linestyle='--', alpha=0.3)
    plt.xlim(t_base[0], t_base[-1])
    plt.legend(loc='upper right')

    plt.figure(2)
    title = 'Cavity Detuning Spectrum\n' + tt
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.plot(xf, 2.0/N * np.abs(fft[0:N//2]), label = label1)
    plt.plot(xf, 2.0/N * np.abs(fft2[0:N//2]), label = label2)
    plt.xlim(0,max_freq)
    plt.legend(loc='upper right')

    plt.figure(3)
    title = 'Cavity Detuning PSD\n' + tt
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Detuning PSD [Hz]')
    plt.semilogy(freq[1:],psd[1:], label = label1)
    plt.semilogy(freq2[1:],psd2[1:], label = label2)
    plt.xlim(0,max_freq)
    plt.legend(loc='upper right')

    plt.figure(4)
    title = 'Detuning Histogram\n' + tt
    plt.title(title)
    plt.xlabel('Detuning [Hz]')
    plt.ylabel('Counts')
    plt.hist(data1, bins=140,  histtype='step', log='True', label=label1)
    plt.hist(data2, bins=140,  histtype='step', log='True', label=label2)
    plt.axvline(x=-10, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=10, color='r', linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')

    plt.figure(5)
    title = 'Cumulative Detuning STD\n' + tt
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Detuning STD [Hz]')
    plt.plot(fftfreq(N, dt)[:N//2], c_d[:N//2], label=label1)
    plt.plot(fftfreq(N, dt)[:N//2], c_d2[:N//2], label=label2)
    print(np.std(data1))
    print(np.std(data2))
    plt.xlim(0, max_freq)
    plt.ylim(bottom=0)
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Microphonics plotter")
    parser.add_argument('-f', '--file', dest='datafile', help='Waveform data file')
    parser.add_argument('-wsp', '--wave_samp_per', dest='wsp', default=1, type=int,
                        help='Waveform decimation factor')
    parser.add_argument('-c', '--conf', dest='conf_file', help='Configuration File')
    parser.add_argument('-t', '--title', dest='tt', help='Plot Title')

    args = parser.parse_args()

    mp_plot(args.datafile, args.wsp, args.conf_file, args.tt)

