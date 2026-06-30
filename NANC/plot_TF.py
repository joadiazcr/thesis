import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq
import re
from datetime import datetime


plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=14)
plt.rc('figure', titlesize=18)


class Wf:
    def __init__(self, raw, dt, count, name, waveform_plot=False):
        self.raw = raw
        self.dt = dt
        self.name = name
        self.count = count
        self.fold()
        self.compute_fft(plot=waveform_plot)

    def fold(self):
        if True:
            self.folded = np.reshape(self.raw, (self.count, -1))
            self.folded = np.mean(self.folded, axis=0)
            self.folded = self.folded - np.mean(self.folded)
        else:
            self.folded = self.raw - np.mean(self.raw)
        self.len = len(self.folded)
        self.df = 1 / (self.dt * self.len)

    def compute_fft(self, plot=False):
        fft_raw = np.fft.fft(self.folded)
        self.fft = fft_raw * self.dt
        self.xf = fftfreq(self.len, self.dt)[:self.len//2]
        if plot:
            fig_raw = plt.figure(num=f'{self.name} FFT')
            plt.figure(fig_raw)
            plt.title(f"{self.name} FFT")
            plt.plot(self.xf, 2.0/self.len * np.abs(self.fft[0:self.len//2]))


class ResData():
    def __init__(self, datafile, count, wsp, waveform_plot=False):
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

        self.p_dac = Wf(self.data[0], self.dt, self.count, 'Piezo DAC', waveform_plot)
        self.detuning = Wf(self.data[1], self.dt, self.count, 'Detuning', waveform_plot)
        self.avdiff = Wf(self.data[2], self.dt, self.count, 'AVDIFF', waveform_plot)
        self.bvdiff = Wf(self.data[3], self.dt, self.count, 'BVDIFF', waveform_plot)
        self.waveforms = [self.p_dac, self.detuning, self.avdiff, self.bvdiff]
        self.tf()

    def plot_raw_data(self, fig, axs, start=0, end=None):
        if end is None:
            end = self.data_len
        fig.suptitle("Raw data")
        y_labels = ['DAC', 'Detuning [Hz]', 'AVDIFF [V]', 'BVDIFF [V]']
        for i in range(self.num_colums):
            axs[i].plot(self.time[start:end], self.data[i][start:end])
            axs[i].set_ylabel(y_labels[i])
        axs[self.num_colums-1].set_xlabel("Time [seconds]")

    def plot_fft(self, fig, axs, start=0, end=None):
        if end is None:
            end = self.data_len
        fig.suptitle("FFTs")
        y_labels = ['DAC', 'Detuning [Hz]', 'AVDIFF [V]', 'BVDIFF [V]']
        for i, wf in enumerate(self.waveforms):
            axs[i].plot(wf.xf, 2.0/wf.len * np.abs(wf.fft[0:wf.len//2]))
            axs[i].set_ylabel(y_labels[i])
        axs[self.num_colums-1].set_xlabel("Frequency [Hz]")

    def tf(self):
        det_fft = self.detuning.fft[0:self.p_dac.len//2]
        #p_dac_fft = self.p_dac.fft[0:self.p_dac.len//2]
        p_dac_fft = self.avdiff.fft[0:self.p_dac.len//2]
        self.tf_real = np.real(det_fft/p_dac_fft)
        self.tf_imag = np.imag(det_fft/p_dac_fft)
        self.tf_mag = np.sqrt(self.tf_real**2 + self.tf_imag**2)
        self.tf_phs = np.unwrap(np.arctan2(self.tf_imag, self.tf_real))

    def plot_tf(self, start, end, fig1, fig2):
        start = int(start/self.p_dac.df)
        end = int(end/self.p_dac.df)
        real = self.tf_real[start:end]
        imag = self.tf_imag[start:end]
        mag = self.tf_mag[start:end]
        freq = self.p_dac.xf[start:end]
        plt.figure(fig1)
        plt.plot(freq, real, label=f'Real {self.datafile.split("/")[-1]}')
        plt.plot(freq, imag, label=f'Imaginary {self.datafile.split("/")[-1]}')

        plt.figure(fig2)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('A [Hz/V]')
        plt.plot(freq, mag, label=f'{self.datafile.split("/")[-1]}')


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Transfer Function plotter")
    parser.add_argument("-f", "--file_list", nargs="+", required=True,
                        help="List of files to stack")
    parser.add_argument('-c', '--count', dest='count', required=True, type=int,
                        help='Number of chirps in the res file')
    parser.add_argument('-wsp', '--wave_samp_per', dest='wsp', default=1,
                        type=int, help='Waveform decimation factor')
    parser.add_argument("-wp", "--waveform_plot", action="store_true",
                        default=False,
                        help="Plot waveform time and frequency domain")
    args = parser.parse_args()

    fig1 = plt.figure(num="TF Real and Imag", figsize=(14, 8))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('A [Hz/V]')
    fig2 = plt.figure(num="TF Magnitude", figsize=(14, 8))
    fig3 = plt.figure(num="Raw data", figsize=(14, 8))
    axs3 = fig3.subplots(4, 1, sharex=True)
    fig4 = plt.figure(num="FFTs", figsize=(14, 8))
    axs4 = fig4.subplots(4, 1, sharex=True)

    for file in args.file_list:
        res_data = ResData(file, args.count, args.wsp, args.waveform_plot)

        filename = file.split('/')[-1]
        pattern = (
            r"^(?P<resolution>res\d+)_"
            r"(?P<signal_type>[a-zA-Z]+)_"
            r"(?P<start_freq>\d+Hz)_"
            r"(?P<end_freq>\d+Hz)_"
            r"(?P<period>per\d+)_"
            r"(?P<wsp>wsp\d+)_"
            r"(?P<date>\d{8})_"
            r"(?P<time>\d{6})$"
        )

        match = re.match(pattern, filename)

        if match:
            data = match.groupdict()
            
            # Optional: Parse the date and time strings into a nice datetime object
            dt_string = f"{data['date']}_{data['time']}"
            data['timestamp'] = datetime.strptime(dt_string, "%Y%m%d_%H%M%S")
            
            # Print the clean results
            print("Parsing Successful!\n")
            for key, value in data.items():
                print(f"{key.upper():<12}: {value}")

            f_start = int(data['start_freq'].removesuffix("Hz"))
            f_end = int(data['end_freq'].removesuffix("Hz"))
        else:
            print("Filename didn't match the expected format.")
            f_start = 50
            f_end = 80

        res_data.plot_raw_data(fig3, axs3)
        res_data.plot_fft(fig4, axs4)
        res_data.plot_tf(f_start, f_end, fig1, fig2)

    
    plt.figure(fig1)
    plt.legend()
    plt.figure(fig2)
    plt.legend()
    plt.show()
