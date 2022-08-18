import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.fft import fftfreq
from scipy import signal


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


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


def mp_plot(data_f, wsp, conf_f, tt):

    data_all = np.loadtxt(data_f)
    N, nch = data_all.shape
    print('data shape: %s X %s' %(N, nch))
    # Assume 2 kHz samp rate
    dt = wsp/2e3
    t_base = np.arange(0, dt*N, dt)
    max_freq = 250

    f = open(conf_f)
    conf = json.load(f)
    exps = []
    labels = []
    for i in conf:
        exps = np.append(exps, i)
        labels = np.append(labels, conf[i])

    for i in conf:
        data = data_all[:, int(i)]
        label = conf[i]
        print("json column %s, label %s" %(i, label))
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
        print("Detuning peak: %s Hz" %peak)
        plt.figure(1)
        if tt != '':
            title = 'Cavity Detuning\n' + tt
            plt.title(title)
        plt.xlabel('Time [s]')
        plt.ylabel('Detuning [Hz]')
        plt.plot(t_base, data, label=label)
        plt.axhline(y=-10, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=10, color='r', linestyle='--', alpha=0.3)
        plt.xlim(t_base[0], t_base[-1])
        plt.legend(loc='upper right')
        plt.tight_layout()

        plt.figure(2)
        if tt != '':
            title = 'Cavity Detuning Spectrum\n' + tt
            plt.title(title)
        plt.xlabel('Frequency [Hz]')
        plt.plot(xf, 2.0/N * np.abs(fft[0:N//2]), label=label)
        plt.xlim(0, max_freq)
        plt.legend(loc='upper right')
        plt.tight_layout()

        plt.figure(3)
        if tt != '':
            title = 'Cavity Detuning PSD\n' + tt
            plt.title(title)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Detuning PSD [Hz]')
        plt.semilogy(freq[1:], psd[1:], label=label)
        plt.xlim(0, max_freq)
        plt.legend(loc='upper right')
        plt.tight_layout()

        plt.figure(4)
        if tt != '':
            title = 'Detuning Histogram\n' + tt
            plt.title(title)
        plt.xlabel('Detuning [Hz]')
        plt.ylabel('Counts')
        plt.hist(data, bins=140,  histtype='step', log='True', label=label)
        plt.axvline(x=-10, color='r', linestyle='--', alpha=0.3)
        plt.axvline(x=10, color='r', linestyle='--', alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()

        plt.figure(5)
        if tt != '':
            title = 'Cumulative Detuning STD\n' + tt
            plt.title(title)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Detuning STD [Hz]')
        plt.plot(fftfreq(N, dt)[:N//2], c_d[:N//2], label=label)
        print("STD: %s" %(np.std(data)))
        plt.xlim(0, max_freq)
        plt.ylim(bottom=0)
        plt.legend(loc='upper right')
        plt.tight_layout()

    plt.figure(6)
    if tt != '':
        title = 'Spectrogram of Detuning\n' + tt
        plt.title(title)
    f, t, Sxx = signal.spectrogram(data_all[:, 1], 1/dt)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Microphonics plotter")
    parser.add_argument('-f', '--file', dest='datafile', required=True,
                        help='Waveform data file')
    parser.add_argument('-wsp', '--wave_samp_per', dest='wsp', default=1,
                        type=int, help='Waveform decimation factor')
    parser.add_argument('-c', '--conf', dest='conf_file', required=True,
                        help='Configuration File')
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

    mp_plot(args.datafile, args.wsp, args.conf_file, args.tt)
