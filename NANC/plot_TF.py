import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq


plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=14)
plt.rc('figure', titlesize=18)


class Wf:
    def __init__(self, raw, dt, count, name):
        self.raw = raw
        self.dt = dt
        self.name = name
        self.count = count
        self.fold()
        self.compute_fft()

    def fold(self):
        self.folded = np.reshape(self.raw, (self.count, -1))
        self.folded = np.mean(self.folded, axis=0)
        self.folded = self.folded - np.mean(self.folded)
        self.len = len(self.folded)

    def compute_fft(self, plot=False):
        fft_raw = np.fft.fft(self.folded)/self.len
        self.fft = fft_raw * self.dt * self.len
        self.xf = fftfreq(self.len, self.dt)[:self.len//2]
        if plot:
            plt.title(f"{self.name}")
            plt.plot(self.xf, 2.0/self.len * np.abs(self.fft[0:self.len//2]))
            plt.show()


class ResData():
    def __init__(self, datafile, count, wsp):
        self.datafile = datafile
        self.count = count
        self.wsp = wsp
        self.dt = self.wsp/2e3  # 2 kHz samp rate when wsp=1

        with open(self.datafile) as fil:
            self.header = fil.readline().strip("# \n")
        self.data = np.loadtxt(self.datafile).T
        self.num_colums, self.data_len = self.data.shape
        print(f'Data shape: {self.num_colums} x {self.data_len}')
        self.time = np.arange(0, self.dt*self.data_len, self.dt)
        self.df = 1 / (self.dt * self.data_len)

        self.p_dac = Wf(self.data[0], self.dt, self.count, 'Piezo DAC')
        self.detuning = Wf(self.data[1], self.dt, self.count, 'Detuning')
        self.tf()

    def plot_raw_data(self, start, end):
        fig, axs = plt.subplots(self.num_colums, 1)
        fig.suptitle("Raw data")
        y_labels = ['DAC', 'Detuning [Hz]', 'AVDIFF [V]', 'BVDIFF [V]']
        for i in range(self.num_colums):
            axs[i].plot(self.time[start:end], self.data[i][start:end])
            axs[i].set_ylabel(y_labels[i])
        axs[self.num_colums-1].set_xlabel("Time [seconds]")
        plt.show()

    def tf(self):
        det_fft = self.detuning.fft[0:self.p_dac.len//2]
        p_dac_fft = self.p_dac.fft[0:self.p_dac.len//2]
        self.tf_real = np.real(det_fft/p_dac_fft)
        self.tf_imag = np.imag(det_fft/p_dac_fft)
        self.tf_mag = np.sqrt(self.tf_real**2 + self.tf_imag**2)
        self.tf_phs = np.unwrap(np.arctan2(self.tf_imag, self.tf_real))

    def plot_tf(self, start, end):
        amp_max = np.max(self.tf_mag[start:end*3])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('A [Hz/V]')
        plt.plot(self.p_dac.xf, self.tf_real, label='Real')
        plt.plot(self.p_dac.xf, self.tf_imag, label='Imaginary')
        plt.xlim(start, end)
        plt.ylim(-1.1 * amp_max, 1.1 * amp_max)
        plt.legend()
        plt.show()

        plt.xlabel('Frequency [Hz]')
        plt.ylabel('A [Hz/V]')
        plt.plot(self.p_dac.xf, self.tf_mag)
        plt.xlim(start, end)
        plt.ylim(0, 1.1 * amp_max)
        plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Transfer Function plotter")
    parser.add_argument('-f', '--file', dest='datafile', required=True,
                        help='Resonance control chassis data file')
    parser.add_argument('-c', '--count', dest='count', required=True, type=int,
                        help='Number of chirps in the res file')
    parser.add_argument('-wsp', '--wave_samp_per', dest='wsp', default=1,
                        type=int, help='Waveform decimation factor')
    args = parser.parse_args()

    res_data = ResData(args.datafile, args.count, args.wsp)

    start = 0
    end = 80
    res_data.plot_raw_data(start, end)
    f_start = 1
    f_end = 80
    res_data.plot_tf(f_start, f_end)
