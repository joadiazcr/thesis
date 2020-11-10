import numpy as np
import matplotlib.pyplot as plt
import argparse
from os.path import dirname, join as pjoin
import scipy.io as sio


def basic_file_info(data):
    H101 = np.mean(data['Magnets'][3,:])
    V101 = np.mean(data['Magnets'][4,:])
    H103 = np.mean(data['Magnets'][5,:])
    V103 = np.mean(data['Magnets'][6,:])
    print('H101 = ', "{:.2f}".format(H101), 'A')
    print('V101 = ', "{:.2f}".format(V101), 'A')
    print('H103 = ', "{:.2f}".format(H103), 'A')
    print('V103 = ', "{:.2f}".format(V103), 'A')
    
    #print(np.mean(data['BPMMags'][1,:,:],0))
    #print(np.max(data['BPMMags'][1,:,:]))
    numBunches = len(np.where(np.mean(data['BPMMags'][1,:,:],0)/np.max(data['BPMMags'][1,:,:])>0.1)[0])
    print('Bunches = ', numBunches)

    bCharge = (np.mean(np.mean(data['Toroids'][0,:,1:]))*100)*10
    print('Bunch charge = ', bCharge,'pC/b')


def plot_toroids(data,mode):
    toroids = data['Toroids']
    num_toroids = toroids.shape[0]
    reps = toroids.shape[1]
    points = toroids.shape[2]
    print('Num toroids = ', num_toroids)
    print('toroids Repetitions = ', reps)
    print('toroids Points = ', points)
    if mode == 0:
        title = 'Toroids'
        for toroid in range(0,num_toroids):
            plt.plot(toroids[toroid][0], label='Toroid %s' %(toroid+1))
            plt.legend()
    else:
        title = 'Toroid ' + str(mode)
        for rep in range(0,reps):
            plt.plot(toroids[mode-1][rep])
    plt.title(title)
    plt.show()


def plot_BPMMags(data,mode):
    BPMMags = data['BPMMags']
    num_BPMMags = BPMMags.shape[0]
    reps = BPMMags.shape[1]
    points = BPMMags.shape[2]
    print('Num BPMMags = ', num_BPMMags)
    print('BPMMags Repetitions = ', reps)
    print('BPMMAgs Points = ', points)
    if mode == 0:
        title = 'BPMMags'
        for bpmMag in range(0,num_BPMMags):
            plt.plot(BPMMags[bpmMag][0])
    else:
        title = 'BPMMag ' + str(mode)
        for rep in range(0,reps):
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
        for hom in range(0,num_HOMs):
            plt.plot(HOMs[hom][0], label='HOM %s' %data['HOMs_List'][hom])
            plt.legend()
    else:
        title = 'HOM ' + str(mode)
        for rep in range(0,reps):
            plt.plot(HOMs[mode-1][rep])
    plt.xlim([740,790])
    plt.title(title)
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
    parser.add_argument('-m', metavar='mode', dest='mode', type=int, choices=[0,1,2,3,4,5,6], default=0,
                    help='Select the device number. 0 to plot all devices')


    args = parser.parse_args()

    data = sio.loadmat(args.data_file)
    HOMsList = ['CC1 Upstream', 'CC1 Downstream', 'CC2 Upstream', 'CC2 Downstream', 'C1IW1A?', 'C1IW1B?']
    data['HOMs_List'] = HOMsList

    basic_file_info(data)
    if args.tor: plot_toroids(data,args.mode)
    if args.bpm: plot_BPMMags(data,args.mode)
    if args.hom: plot_raw_HOMs(data,args.mode)
