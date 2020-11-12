import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.io as sio
import csv


def basic_file_info(data):
    H101 = np.mean(data['Magnets'][3, :])
    V101 = np.mean(data['Magnets'][4, :])
    H103 = np.mean(data['Magnets'][5, :])
    V103 = np.mean(data['Magnets'][6, :])
    print('H101 = ', "{:.2f}".format(H101), 'A')
    print('V101 = ', "{:.2f}".format(V101), 'A')
    print('H103 = ', "{:.2f}".format(H103), 'A')
    print('V103 = ', "{:.2f}".format(V103), 'A')

    # print(np.mean(data['BPMMags'][1,:,:],0))
    # print(np.max(data['BPMMags'][1,:,:]))
    m = np.mean(data['BPMMags'][1, :, :], 0)
    mx = np.max(data['BPMMags'][1, :, :])
    numBunches = len(np.where(m/mx > 0.1)[0])
    print('Bunches = ', numBunches)

    bCharge = (np.mean(np.mean(data['Toroids'][0, :, 1:]))*100)*10
    print('Bunch charge = ', bCharge, 'pC/b')


def plot_toroids(data, mode):
    toroids = data['Toroids']
    num_toroids = toroids.shape[0]
    reps = toroids.shape[1]
    points = toroids.shape[2]
    print('Num toroids = ', num_toroids)
    print('toroids Repetitions = ', reps)
    print('toroids Points = ', points)
    if mode == 0:
        title = 'Toroids'
        for toroid in range(0, num_toroids):
            plt.plot(toroids[toroid][0], label='Toroid %s' % (toroid + 1))
            plt.legend()
    else:
        title = 'Toroid ' + str(mode)
        for rep in range(0, reps):
            plt.plot(toroids[mode-1][rep])
    plt.title(title)
    plt.show()


def plot_BPMMags(data, mode):
    BPMMags = data['BPMMags']
    num_BPMMags = BPMMags.shape[0]
    reps = BPMMags.shape[1]
    points = BPMMags.shape[2]
    print('Num BPMMags = ', num_BPMMags)
    print('BPMMags Repetitions = ', reps)
    print('BPMMAgs Points = ', points)
    if mode == 0:
        title = 'BPMMags'
        for bpmMag in range(0, num_BPMMags):
            plt.plot(BPMMags[bpmMag][0])
    else:
        title = 'BPMMag ' + str(mode)
        for rep in range(0, reps):
            plt.plot(BPMMags[mode-1][rep])
    plt.title(title)
    plt.show()


def plot_raw_HOMs(data, mode):
    HOMs = data['HOMs']
    num_HOMs = HOMs.shape[0]
    reps = HOMs.shape[1]
    points = HOMs.shape[2]
    print('Num HOMs = ', num_HOMs)
    print('Repetitions = ', reps)
    print('Points = ', points)
    if mode == 0:
        title = 'HOMs'
        for hom in range(0, num_HOMs):
            plt.plot(HOMs[hom][0], label='HOM %s' % data['HOMs_List'][hom])
            plt.legend()
    else:
        title = 'HOM ' + str(mode)
        for rep in range(0, reps):
            plt.plot(HOMs[mode-1][rep])
    plt.xlim([740, 790])
    plt.title(title)
    plt.show()


def process_HOM_data(data):
    HOMs = data['HOMs']
    num_HOMs = HOMs.shape[0]
    reps = HOMs.shape[1]
    reps=26

    fig, axs = plt.subplots(2, 2)
    homs_peak_based_mean = np.zeros(num_HOMs)
    homs_peak_based_std = np.zeros(num_HOMs)
    for hom in range(0,num_HOMs-2):
        homs_peaks_based = np.zeros(reps)
        for rep in range(0,reps):
            hom_baseln = np.mean(np.concatenate((HOMs[hom][rep][400:600], HOMs[hom][rep][900:1000])))
            hom_based = HOMs[hom][rep] - hom_baseln
            axs[int(hom/2), hom%2].plot(hom_based)
            homs_peaks_based[rep] = np.min(hom_based)
        homs_peak_based_mean[hom] = np.mean(homs_peaks_based)
        homs_peak_based_std[hom] = np.std(homs_peaks_based)
        axs[int(hom/2), hom%2].set_xlim(740, 810)

    print('homs_peak_based_mean = ', homs_peak_based_mean)
    print('homs_peak_based_std = ', homs_peak_based_std)

    fig.suptitle('225 pC, 50 b, 0 Ampl, V101 = 1.04 A \n AllData_2020-11-06-19-27-15.mat', fontsize=16) # Hardcoded title
    axs[0,0].set_title('CC1 Upstr Peak_Mean = %.3f Peak_STD = %.4f' % (homs_peak_based_mean[0], homs_peak_based_std[0]))
    axs[0,1].set_title('CC1 Dwnstr Peak_Mean = %.3f Peak_STD = %.4f' % (homs_peak_based_mean[1], homs_peak_based_std[1]))
    axs[1,0].set_title('CC2 Upstr Peak_Mean = %.3f Peak_STD = %.4f' % (homs_peak_based_mean[2], homs_peak_based_std[2]))
    axs[1,1].set_title('CC2 Dwnstr Peak_Mean = %.3f Peak_STD = %.4f' % (homs_peak_based_mean[3], homs_peak_based_std[3]))
    plt.legend()
    plt.show()


def V101_HOM(data_file):
    with open(data_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        v101 = []
        hom1 = []
        hom2 = []
        hom3 = []
        hom4 = []
        hom1_std = []
        hom2_std = []
        hom3_std = []
        hom4_std = []
        for row in readCSV:
            if 'V101' not in row:
                v101.append(float(row[0]))
                hom1.append(float(row[1]))
                hom2.append(float(row[2]))
                hom3.append(float(row[3]))
                hom4.append(float(row[4]))
                hom1_std.append(float(row[5]))
                hom2_std.append(float(row[6]))
                hom3_std.append(float(row[7]))
                hom4_std.append(float(row[8]))

    plt.errorbar(v101,hom1,yerr=hom1_std,label='CC1 Upstr')
    plt.errorbar(v101,hom2,yerr=hom2_std,label='CC1 Dwnstr')
    plt.errorbar(v101,hom3,yerr=hom3_std,label='CC2 Upstr')
    plt.errorbar(v101,hom4,yerr=hom4_std,label='CC2 Dwnstr')
    plt.xlabel('V101 [A]')
    plt.ylabel('HOM Signal')
    plt.title('225 pC, 50b, 0Ampl') # Hardcoded title
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    des = 'Procces and plot FAST data'
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('-f', '--file', dest='data_file', required=True,
                        help='File with data to be processed/plotted')
    parser.add_argument('-hom', action="store_true",
                        help='Plot HOM data')
    parser.add_argument('-bpm', action="store_true",
                        help='Plot BPM data')
    parser.add_argument('-tor', action="store_true",
                        help='Plot toroid data')
    parser.add_argument('-p', action="store_true",
                        help='Process HOM data')
    parser.add_argument('-vh', action="store_true",
                        help='Plot V101 vs HOMs')
    parser.add_argument('-m', metavar='mode', dest='mode', type=int,
                        choices=range(0, 7), default=0,
                        help='Select the device number. 0 to plot all devices')

    args = parser.parse_args()

    if args.vh:
        V101_HOM(args.data_file)
        exit()

    data = sio.loadmat(args.data_file)
    HOMsList = ['CC1 Upstream', 'CC1 Downstream', 'CC2 Upstream',
                'CC2 Downstream', 'C1IW1A?', 'C1IW1B?']
    data['HOMs_List'] = HOMsList
    basic_file_info(data)
    if args.p:
        process_HOM_data(data)
    if args.tor:
        plot_toroids(data, args.mode)
    if args.bpm:
        plot_BPMMags(data, args.mode)
    if args.hom:
        plot_raw_HOMs(data, args.mode)