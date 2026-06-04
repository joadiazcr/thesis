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

    def compute_fft(self):
        fft_raw = np.fft.fft(self.folded)/self.len
        self.fft = fft_raw * self.dt * self.len
        self.xf = fftfreq(self.len, self.dt)[:self.len//2]
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

        self.compute_fft()

    def compute_fft(self):
        self.fft = np.zeros((self.num_colums, self.data_len), dtype=complex)
        for i in range(self.num_colums):
            print(self.data[i])
            fft_raw = np.fft.fft(self.data[i])/self.data_len
            self.fft[i] = fft_raw * self.dt * self.data_len
        self.xf = fftfreq(self.data_len, self.dt)[:self.data_len//2]

    def plot_raw_data(self, start, end):
        fig, axs = plt.subplots(self.num_colums, 1)
        fig.suptitle("Raw data")
        y_labels = ['DAC', 'Detuning [Hz]', 'AVDIFF [V]', 'BVDIFF [V]']
        for i in range(self.num_colums):
            axs[i].plot(self.time[start:end], self.data[i][start:end])
            axs[i].set_ylabel(y_labels[i])
        axs[self.num_colums-1].set_xlabel("Time [seconds]")
        plt.show()

    def plot_fft(self):
        fig, axs = plt.subplots(self.num_colums, 1)
        y_labels = ['DAC', 'Detuning [Hz]', 'AVDIFF [V]', 'BVDIFF [V]']
        for i in range(4):
            axs[i].plot(self.xf,
                        2.0/self.data_len * np.abs(self.fft[i][0:self.data_len//2]))
            axs[i].set_ylabel(y_labels[i])
        axs[self.num_colums-1].set_xlabel("Frequency [Hz]")
        plt.show()

    def plot_tf(self, start, end):
        sf = int(start/self.df)
        ef = int(end/self.df)
        plt.plot(self.xf[sf:ef], np.real(self.fft[1][0:self.data_len//2]/self.fft[0][0:self.data_len//2])[sf:ef])
        plt.plot(self.xf[sf:ef], np.imag(self.fft[1][0:self.data_len//2]/self.fft[0][0:self.data_len//2])[sf:ef])
        plt.show()

        plt.plot(self.p_dac.xf, np.real(self.detuning.fft[0:self.p_dac.len//2]/self.p_dac.fft[0:self.p_dac.len//2]))
        plt.plot(self.p_dac.xf, np.imag(self.detuning.fft[0:self.p_dac.len//2]/self.p_dac.fft[0:self.p_dac.len//2]))
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
    end = res_data.data_len
    res_data.plot_raw_data(start, end)
    res_data.plot_fft()
    f_start = 20
    f_end = 300
    res_data.plot_tf(f_start, f_end)
    