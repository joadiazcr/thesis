import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.io as sio
import csv
import pandas as pd
import seaborn as sns
from scipy.integrate import trapz 


date = 'AllData_2021-02-18-'
shift = 6
#csv_file = "HOM_FAST/shift3_20201120/cmhoms_peaks.csv"
#csv_file = "HOM_FAST_20201106/shift4_20201203/test.csv"
csv_file = "HOM_FAST_20201106/shift6_20210218/cmhoms_peaks.csv"
main_shots = 300
main_charge = 250
main_bunches = 50
main_amp = 2
V125 = 4.343
H125 = 0.978
main_time = '18-45-27'
slac_loc = "SLAC Chassis Upstr"
fnal_loc = "FNAL Chassis Dwnstr"

file_to_process = date + main_time + '.mat'
main_title = ('CM2 ' + str(main_amp) + ' amplifier, ' + str(main_charge) + ' pC, ' + str(main_bunches) + 'b, V125=' + str(V125) + ' H125=' + str(H125) + ', 1st shot\n' +
               file_to_process)
main_title2 = ('CM2 ' + str(main_amp) + ' amplifier, ' + str(main_charge) + ' pC, ' + str(main_bunches) + ' b, V125=' + str(V125) + ' H125=' + str(H125) + ', ' + str(main_shots) + ' shots\n' +
               file_to_process)


def basic_file_info(data):
    print(data.keys())
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
    #m = np.mean(data['BPMMags'][1, :, :], 0)
    #mx = np.max(data['BPMMags'][1, :, :])
    #numBunches = len(np.where(m/mx > 0.1)[0])
    #print('Bunches = ', numBunches)

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


def plot_BPM(data, mode, bpm_type):
    print("Plotting %s data" % bpm_type)
    BPMs = data[bpm_type]
    num_BPMs = BPMs.shape[0]
    reps = BPMs.shape[1]
    points = BPMs.shape[2]
    print('Num BPMs = ', num_BPMs)
    print('BPMs Repetitions = ', reps)
    print('BPMs Points = ', points)
    if mode == 0:
        title = 'BPMs'
        for bpm in range(0, num_BPMs):
            plt.plot(BPMs[bpm][0], label='BPM %s' %(bpm+1))
        plt.legend()
    else:
        title = 'BPM ' + str(mode)
        for rep in range(0, reps):
            plt.plot(BPMs[mode-1][rep])
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_raw_HOMs(data, mode):
    HOMs = data['HOMs']
    num_HOMs = HOMs.shape[0]
    reps = HOMs.shape[1]
    points = HOMs.shape[2] # num of bunches + 6
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
    plt.xlim([740, 810])
    plt.title(title)
    plt.show()


def plot_raw_CMHOMs(data, mode, s):
    CMHOMs = data['CMHOMs']
    num_CMHOMs = CMHOMs.shape[0]
    reps = CMHOMs.shape[1]
    points = CMHOMs.shape[2] # num of bunches + 6
    print('Num CMHOMs = ', num_CMHOMs)
    print('Repetitions = ', reps)
    print('Points = ', points)
    if mode == 0:
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(main_title, fontsize=16)  # Hardcoded title
        for cmhom in range(0, 8):
            axs[0].plot(CMHOMs[cmhom][0], label=data['CMHOMs_List'][cmhom])
            axs[0].legend()
            axs[0].set_xlabel('Sample')
            axs[0].set_ylabel('HOM Signal (V)')
            axs[0].set_title(slac_loc)
            axs[0].set_xlim(200,750)
        for cmhom in range(8, num_CMHOMs):
            axs[1].plot(CMHOMs[cmhom][0], label=data['CMHOMs_List'][cmhom])
            axs[1].legend()
            axs[1].set_xlabel('Sample')
            axs[1].set_ylabel('HOM Signal (V)')
            axs[1].set_title(fnal_loc)
            axs[1].set_xlim(200,750)

        fig2, axs2 = plt.subplots(figsize=(7, 6))
        for cmhom in range(0, 8):
            time = (np.arange(0, len(CMHOMs[cmhom][0]), 1) - 200) * 0.125 # 8MHz sample rate. T=0.125 us
            axs2.plot(time, CMHOMs[cmhom][0], label=data['CMHOMs_List'][cmhom])
            axs2.legend(fontsize=12)
            axs2.set_xlabel('Time ' + r'$[\mu s]$', fontsize=14)
            axs2.set_ylabel('HOM Signal (V)', fontsize=14)
            axs2.set_xlim(0,550 * 0.125)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

    else:
        title = 'CMHOM ' + str(mode)
        fig, axs = plt.subplots(1, 1)
        fig.suptitle(title, fontsize=16)  # Hardcoded title
        for rep in range(0, reps):
            axs.plot(CMHOMs[mode-1][rep])

    fig.set_size_inches((15.5, 8))
    if s:
        fig.savefig(main_time + '_first.png', dpi=500)
    plt.show()

def process_BPM_data(data, bpm):
    print("Processing BPM data")
    BPMs = data[bpm]
    num_BPMs = BPMs.shape[0]
    reps = BPMs.shape[1]
    points = BPMs.shape[2]
    print('Num BPMs = ', num_BPMs)
    print('BPMs Repetitions = ', reps)
    print('BPMs Points = ', points)

    BPMstart = 10
    BPMend = 50

    bpm_means = np.zeros(num_BPMs)
    #for i in range(0,num_BPMs):
    for i in range(0,1):
        bpm_rep_mean = np.zeros(reps)
        for rep in range(0,reps):
            bpm_rep_mean[rep] = np.mean(BPMs[i][rep][BPMstart:BPMend])
            print(bpm_rep_mean[rep])
        exit()
        bpm_means[i] = np.mean(bpm_rep_mean)
    print(bpm_means)

    if bpm == "BPMs":
        cc1_pos = 2.79763  # meters
        cc2_pos = 5.506526  # meters
        ccLen = 1.0  # meters

        bpm_means_h = bpm_means[0:][::2] # Horizontal BPMs
        bpm_means_h = [x / 1000 for x in bpm_means_h]
        bpm_means_v = bpm_means[1:][::2] # Vertical BPMs
        bpm_means_v = [x / 1000 for x in bpm_means_v]
        plt.plot(data['bpmPos'][0:6], bpm_means_h[0:6], 'o-')
        plt.xlabel('Beamline Position (m)')
        plt.ylabel('Position (mm)')
        plt.xlim([0, 10])
        plt.ylim([-15, 15])
        plt.vlines(cc1_pos - (ccLen/2.0), -15, 15)
        plt.vlines(cc1_pos + (ccLen/2.0), -15, 15)
        plt.vlines(cc2_pos - (ccLen/2.0), -15, 15)
        plt.vlines(cc2_pos + (ccLen/2.0), -15, 15)
        plt.text(cc1_pos , -14, 'CC1', ha='center')
        plt.text(cc2_pos, -14, 'CC2', ha='center')
        plt.show()


def process_HOM_data(data):
    HOMs = data['HOMs']
    num_HOMs = HOMs.shape[0]
    reps = HOMs.shape[1]

    fig, axs = plt.subplots(2, 2)
    homs_pk_bsd_mean = np.zeros(num_HOMs)
    homs_pk_bsd_std = np.zeros(num_HOMs)
    for hom in range(0, num_HOMs-2):
        homs_pks_bsd = np.zeros(reps)
        for rep in range(0, reps):
            hom_baseln_a = HOMs[hom][rep][400:600]
            hom_baseln_b = HOMs[hom][rep][900:1000]
            hom_baseln = np.mean(np.concatenate((hom_baseln_a, hom_baseln_b)))
            hom_based = HOMs[hom][rep] - hom_baseln
            axs[int(hom / 2), hom % 2].plot(hom_based)
            homs_pks_bsd[rep] = np.min(hom_based)
        homs_pk_bsd_mean[hom] = np.mean(homs_pks_bsd)
        homs_pk_bsd_std[hom] = np.std(homs_pks_bsd)
        axs[int(hom / 2), hom % 2].set_xlim(740, 810)

    print('homs_peak_based_mean = ', homs_pk_bsd_mean)
    print('homs_peak_based_std = ', homs_pk_bsd_std)

    txt = ('SLAC #2 chassis, 100 pC, 1 b, 1 Ampl, H101 = -0.875 A, 100 shots\n' +
           'AllData_2020-11-12-21-18-11.mat')
    fig.suptitle(txt, fontsize=16)  # Hardcoded title
    txt = ('CC1 Upstr Peak_Mean = ' + "{:.3f}".format(homs_pk_bsd_mean[0]) +
           '  Peak_STD = ' + "{:.4f}".format(homs_pk_bsd_std[0]))
    axs[0, 0].set_title(txt)
    txt = ('CC1 Dwnstr Peak_Mean = ' + "{:.3f}".format(homs_pk_bsd_mean[1]) +
           '  Peak_STD = ' + "{:.4f}".format(homs_pk_bsd_std[1]))
    axs[0, 1].set_title(txt)
    txt = ('CC2 Upstr Peak_Mean = ' + "{:.3f}".format(homs_pk_bsd_mean[2]) +
           '  Peak_STD = ' + "{:.4f}".format(homs_pk_bsd_std[2]))
    axs[1, 0].set_title(txt)
    txt = ('CC2 Dwnstr Peak_Mean = ' + "{:.3f}".format(homs_pk_bsd_mean[3]) +
           '  Peak_STD = ' + "{:.4f}".format(homs_pk_bsd_std[3]))
    axs[1, 1].set_title(txt)
    plt.legend()
    plt.show()


def process_CMHOM_data(data, s):
    CMHOMs = data['CMHOMs']
    num_CMHOMs = CMHOMs.shape[0]
    reps = CMHOMs.shape[1]

    fig1, axs1 = plt.subplots(2, 4)
    plt.subplots_adjust(left=0.06, bottom=0.07, right=0.97, top=None, wspace=None, hspace=0.24)
    fig1.suptitle(slac_loc + ' ' + main_title2, fontsize=16)  # Hardcoded title
    fig2, axs2 = plt.subplots(2, 3)
    plt.subplots_adjust(left=0.06, bottom=0.07, right=0.97, top=None, wspace=None, hspace=0.24)
    fig2.suptitle(fnal_loc + ' ' + main_title2, fontsize=16)  # Hardcoded title
    cmhoms_pk_bsd_mean = np.zeros(num_CMHOMs)
    cmhoms_pk_bsd_std = np.zeros(num_CMHOMs)
    for cmhom in range(0, num_CMHOMs):
        cmhoms_pks_bsd = np.zeros(reps)
        for rep in range(0, reps):
            cmhom_baseln_a = CMHOMs[cmhom][rep][100:200]
            cmhom_baseln_b = CMHOMs[cmhom][rep][1000:1500]
            cmhom_baseln = np.mean(np.concatenate((cmhom_baseln_a, cmhom_baseln_b)))
            cmhom_based = CMHOMs[cmhom][rep] - cmhom_baseln
            if cmhom < 12:
                cmhoms_pks_bsd[rep] = np.max(cmhom_based)
            elif cmhom >=12:
                cmhoms_pks_bsd[rep] = np.min(cmhom_based)
            if cmhom < 8:
                axs1[int(cmhom / 4), cmhom % 4].plot(cmhom_based)
            elif cmhom >= 8:
                axs2[int((cmhom-8) / 3), (cmhom-8) % 3].plot(cmhom_based)
        cmhoms_pk_bsd_mean[cmhom] = np.mean(cmhoms_pks_bsd)
        cmhoms_pk_bsd_std[cmhom] = np.std(cmhoms_pks_bsd)
        txt = (data['CMHOMs_List'][cmhom] + ' Pk_mean = ' + "{:.3f}".format(cmhoms_pk_bsd_mean[cmhom]) +
               '  Pk_STD = ' + "{:.4f}".format(cmhoms_pk_bsd_std[cmhom]))
        axs_title = txt
        if cmhom < 8:
            axs1[int(cmhom / 4), cmhom % 4].set_xlim(200, 1000)
            axs1[int(cmhom / 4), cmhom % 4].set_title(axs_title)
            if int(cmhom / 4) == 1:
                axs1[int(cmhom / 4), cmhom % 4].set_xlabel('Sample')
            if cmhom % 4 == 0:
                axs1[int(cmhom / 4), cmhom % 4].set_ylabel('HOM Signal (V)')
        elif cmhom >= 8:
            axs2[int((cmhom-8) / 3), (cmhom-8) % 3].set_xlim(200, 750)
            axs2[int((cmhom-8) / 3), (cmhom-8) % 3].set_title(axs_title)
            if int((cmhom-8) / 3) == 1:
                axs2[int((cmhom-8) / 3), (cmhom-8) % 3].set_xlabel('Sample')
            if (cmhom-8) % 3 == 0:
                axs2[int((cmhom-8) / 3), (cmhom-8) % 3].set_ylabel('HOM Signal (V)')

    print('cmhoms_peak_based_mean = ')
    to_print=''
    for x in cmhoms_pk_bsd_mean:
        to_print = to_print + ',' + str(x)
    print(to_print)
    print('cmhoms_peak_based_std = ')
    to_print=''
    for x in cmhoms_pk_bsd_std:
        to_print = to_print + ',' + str(x)
    print(to_print)

    fig1.set_size_inches((15.5, 8))
    fig2.set_size_inches((15.5, 8))
    if False: # if s:
        fig1.savefig(main_time + '_SLAC.png', dpi=500)
        fig2.savefig(main_time + '_FNAL.png', dpi=500)  
           
        # save to csv file
        data = pd.read_csv(csv_file)
        new_row = [shift,main_time,main_bunches,main_charge,main_shots,V125,H125,slac_loc] # time,bunches,pC_b,shots
        new_row = np.append(new_row,cmhoms_pk_bsd_mean)
        new_row = pd.DataFrame([new_row], columns=data.columns)
        data = data.append(new_row, ignore_index=True)
        data.to_csv(csv_file,index=False,)      
        
    plt.legend()
    plt.show()






def integral_CMHOM(data, s):
    x = np.linspace(0, np.pi, 50)
    y = np.sin(x)
    int = trapz(y,x)
    print(int)
    plt.plot(x,y)
    plt.show()
    exit()
    
    CMHOMs = data['CMHOMs']
    num_CMHOMs = CMHOMs.shape[0]
    reps = CMHOMs.shape[1]

    cmhoms_int_bsd_mean = np.zeros(num_CMHOMs)
    cmhoms_int_bsd_std = np.zeros(num_CMHOMs)
    for cmhom in range(0, num_CMHOMs):
        cmhoms_int_bsd = np.zeros(reps)
        for rep in range(0, reps):
            cmhom_baseln_a = CMHOMs[cmhom][rep][100:200]
            cmhom_baseln_b = CMHOMs[cmhom][rep][1000:1500]
            cmhom_baseln = np.mean(np.concatenate((cmhom_baseln_a, cmhom_baseln_b)))
            cmhom_based = CMHOMs[cmhom][rep] - cmhom_baseln
            if cmhom < 12:
                cmhoms_int_bsd[rep] = np.max(cmhom_based) #trapz(cmhom_based,x)
            elif cmhom >=12:
                cmhoms_int_bsd[rep] = np.min(cmhom_based)
        cmhoms_int_bsd_mean[cmhom] = np.mean(cmhoms_int_bsd)
        cmhoms_int_bsd_std[cmhom] = np.std(cmhoms_int_bsd)

    print('cmhoms_peak_based_mean = ')
    to_print=''
    for x in cmhoms_int_bsd_mean:
        to_print = to_print + ',' + str(x)
    print(to_print)
    print('cmhoms_peak_based_std = ')
    to_print=''
    for x in cmhoms_int_bsd_std:
        to_print = to_print + ',' + str(x)
    print(to_print)









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
            if 'V101' not in row and 'h101' not in row:
                v101.append(float(row[0]))
                hom1.append(float(row[1]))
                hom2.append(float(row[2]))
                hom3.append(float(row[3]))
                hom4.append(float(row[4]))
                hom1_std.append(float(row[5]))
                hom2_std.append(float(row[6]))
                hom3_std.append(float(row[7]))
                hom4_std.append(float(row[8]))

    plt.errorbar(v101, np.abs(hom1), yerr=hom1_std, label='CC1 Upstr')
    plt.errorbar(v101, np.abs(hom2), yerr=hom2_std, label='CC1 Dwnstr')
    plt.errorbar(v101, np.abs(hom3), yerr=hom3_std, label='CC2 Upstr')
    plt.errorbar(v101, np.abs(hom4), yerr=hom4_std, label='CC2 Dwnstr')
    plt.xlabel('H101 Magnet Current (A)')
    plt.ylabel('HOM Signal Peak (|V|)')
    plt.title('SLAC Chassis #2, 100 pC, 1 b, 1 Ampl, 100 shots')  # Hardcoded title
    plt.legend()
    plt.grid(True)
    plt.show()


def CMHOM_125(data_file):
    df = pd.read_csv(data_file)
    cavs = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']

    # For shift #6
    #for loc in ['Upstr', 'Dwnstr']:
    #    for bc in [125,250,400,600]:
    #        print("Plotting " + loc + str(bc) + " pB/b")
    #        plt.figure(figsize=(7, 6))
    #        plt.rc('font', weight='bold')
    #        data_frame = df[df['H125']==0.0][df['slac_loc']==loc][df['pC_b']==bc][df['bunches']==50]
    #        data_frame = data_frame.sort_values(by=['V125'])
    #        for cav in cavs:
    #            hom = data_frame[cav+'_mean'].to_numpy() * np.power(10,data_frame['slac_atten']*2/20)
    #            plt.plot(data_frame['V125'].to_numpy(), hom, '--o',label=cav)
    #        plt.ylim(bottom=0.0)
    #        plt.xlabel('V125 Magnet Current from reference (A)', fontsize=16, fontweight='bold')
    #        plt.ylabel('HOM Signal Peak (|V|)', fontsize=16, fontweight='bold')
    #        #plt.title('Downstream, 600 pC, 50 b, 1 Ampl, H125=0.978 A (reference)', fontsize=20)  # Hardcoded title
    #        plt.legend(loc=9, fontsize=14)
    #        plt.tick_params(labelsize=14)
    #        plt.locator_params(axis="x", nbins=7)
    #        plt.grid(True)
    #        plt.show()
    #exit()

    # For shift 3 & 4
    for magnet in ['V125', 'H125']:
        for loc in ['Dwstr', 'Upstr']:
            for bunches in [50,1]:
                plt.figure(figsize=(7, 6))
                plt.rc('font', weight='bold')
                plot_name = magnet + "_CMHOM_" + loc + "_250_" + str(bunches) + ".png"
                print("File name: " + plot_name)

                ref = 4.3 if magnet == 'H125' else 0.96
                ref_mag = 'V125' if magnet =='H125' else 'H125'
                locc = 'SLAC Chassis Dwstr' if loc == 'Dwstr' else 'SLAC Chassis Upstr'
                data = df[df[ref_mag]==ref][df['slac_loc']==locc][df['pC_b']==250][df['bunches']==bunches]
                data = data.sort_values(by=[magnet])
                for cav in cavs:
                    try:
                        plt.plot(data[magnet].to_numpy(), data[cav].to_numpy(), '--o',label=cav)
                    except:
                        print("No data")
                plt.ylim(bottom=0.0)
                plt.xlabel(magnet + ' Magnet Current (A)', fontsize=16, fontweight='bold')
                plt.ylabel('HOM Signal Peak (|V|)', fontsize=16, fontweight='bold')
                #plt.title('SLAC Chassis Downstream, 250 pC, 50 b, 1 Ampl, V125=4.3 A', fontsize=20)  # Hardcoded title
                plt.legend(fontsize=14)
                plt.tick_params(labelsize=14)
                plt.locator_params(axis="x", nbins=7)
                plt.grid(True)
                plt.show()


    exit()

    h_dwstr_250_50 = df[df['V125']==4.3][df['slac_loc']=='SLAC Chassis Dwstr'][df['pC_b']==250][df['bunches']==50]
    h_dwstr_250_50 = h_dwstr_250_50.sort_values(by=['H125'])
    for cav in cavs:
        plt.plot(h_dwstr_250_50['H125'].to_numpy(), h_dwstr_250_50[cav].to_numpy(), '--o',label=cav)
    plt.ylim(bottom=0.0)
    plt.xlabel('H125 Magnet Current (A)', fontsize=20)
    plt.ylabel('HOM Signal Peak (|V|)', fontsize=20)
    plt.title('SLAC Chassis Downstream, 250 pC, 50 b, 1 Ampl, V125=4.3 A', fontsize=20)  # Hardcoded title
    plt.legend(loc='upper right', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.grid(True)
    plt.show()

    h_upstr_250_1 = df[df['V125']==4.3][df['slac_loc']=='SLAC Chassis Upstr'][df['pC_b']==250][df['bunches']==1]
    h_upstr_250_1 = h_upstr_250_1.sort_values(by=['H125'])
    for cav in cavs:
        plt.plot(h_upstr_250_1['H125'].to_numpy(), h_upstr_250_1[cav].to_numpy(), '--o',label=cav)
    plt.ylim(bottom=0.0)
    plt.xlabel('H125 Magnet Current (A)', fontsize=20)
    plt.ylabel('HOM Signal Peak (|V|)', fontsize=20)
    plt.title('SLAC Chassis Upstream, 250 pC, 1 b, 1 Ampl, V125=4.3 A', fontsize=20)  # Hardcoded title
    plt.legend(loc='upper right', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.grid(True)
    plt.show()

    h_dwstr_250_1 = df[df['V125']==4.3][df['slac_loc']=='SLAC Chassis Dwstr'][df['pC_b']==250][df['bunches']==1]
    h_dwstr_250_1 = h_dwstr_250_1.sort_values(by=['H125'])
    for cav in cavs:
        plt.plot(h_dwstr_250_1['H125'].to_numpy(), h_dwstr_250_1[cav].to_numpy(), '--o',label=cav)
    plt.ylim(bottom=0.0)
    plt.xlabel('H125 Magnet Current (A)', fontsize=20)
    plt.ylabel('HOM Signal Peak (|V|)', fontsize=20)
    plt.title('SLAC Chassis Downstream, 250 pC, 1 b, 1 Ampl, V125=4.3 A', fontsize=20)  # Hardcoded title
    plt.legend(loc='upper right', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.grid(True)
    plt.show()

    v_dwstr_250_50 = df[df['H125']==0.96][df['slac_loc']=='SLAC Chassis Dwstr'][df['pC_b']==250][df['bunches']==50]
    v_dwstr_250_50 = v_dwstr_250_50.sort_values(by=['V125'])
    for cav in cavs:
        plt.plot(v_dwstr_250_50['V125'].to_numpy(), v_dwstr_250_50[cav].to_numpy(), '--o',label=cav)
    plt.ylim(bottom=0.0)
    plt.xlabel('V125 Magnet Current (A)', fontsize=20)
    plt.ylabel('HOM Signal Peak (|V|)', fontsize=20)
    plt.title('SLAC Chassis Downstream, 250 pC, 50 b, 1 Ampl, H125=0.96 A', fontsize=20)  # Hardcoded title
    plt.legend(loc='upper right', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.grid(True)
    plt.show()

    v_upstr_250_50 = df[df['H125']==0.96][df['slac_loc']=='SLAC Chassis Upstr'][df['pC_b']==250][df['bunches']==50]
    v_upstr_250_50 = v_upstr_250_50.sort_values(by=['V125'])
    for cav in cavs:
        plt.plot(v_upstr_250_50['V125'].to_numpy(), v_upstr_250_50[cav].to_numpy(), '--o',label=cav)
    plt.ylim(bottom=0.0)
    plt.xlabel('V125 Magnet Current (A)', fontsize=20)
    plt.ylabel('HOM Signal Peak (|V|)', fontsize=20)
    plt.title('SLAC Chassis Upstream, 250 pC, 50 b, 1 Ampl, H125=0.96 A', fontsize=20)  # Hardcoded title
    plt.legend(loc='upper right', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.grid(True)
    plt.show()


def V101_BPM(data_file):
    with open(data_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        v101 = []
        bpm101PH = []
        bpm101PV = []
        bpm102PH = []
        bpm102PV = []
        bpm103PH = []
        bpm103PV = []
        bpm104PH = []
        bpm104PV = []
        bpm106PH = []
        bpm106PV = []
        bpm107PH = []
        bpm107PV = []
        bpm111PH = []
        bpm111PV = []
        bpm113PH = []
        bpm113PV = []
        bpm117PH = []
        bpm117PV = []
        bpm118PH = []
        bpm118PV = []
        bpm120PH = []
        bpm120PV = []
        bpm121PH = []
        bpm121PV = []
        bpm122PH = []
        bpm122PV = []
        bpm123PH = []
        bpm123PV = []
        bpm124PH = []
        bpm124PV = []
        bpm125PH = []
        bpm125PV = []

        for row in readCSV:
            if 'V101' not in row and 'h101' not in row:
                v101.append(float(row[0]))
                bpm101PH.append(float(row[1]))
                bpm101PV.append(float(row[2]))
                bpm102PH.append(float(row[3]))
                bpm102PV.append(float(row[4]))
                bpm103PH.append(float(row[5]))
                bpm103PV.append(float(row[6]))
                bpm104PH.append(float(row[7]))
                bpm104PV.append(float(row[8]))
                bpm106PH.append(float(row[9]))
                bpm106PV.append(float(row[10]))
                bpm107PH.append(float(row[11]))
                bpm107PV.append(float(row[12]))
                bpm111PH.append(float(row[13]))
                bpm111PV.append(float(row[14]))
                bpm113PH.append(float(row[15]))
                bpm113PV.append(float(row[16]))
                bpm117PH.append(float(row[17]))
                bpm117PV.append(float(row[18]))
                bpm118PH.append(float(row[19]))
                bpm118PV.append(float(row[20]))
                bpm120PH.append(float(row[21]))
                bpm120PV.append(float(row[22]))
                bpm121PH.append(float(row[23]))
                bpm121PV.append(float(row[24]))
                bpm122PH.append(float(row[25]))
                bpm122PV.append(float(row[26]))
                bpm123PH.append(float(row[27]))
                bpm123PV.append(float(row[28]))
                bpm124PH.append(float(row[29]))
                bpm124PV.append(float(row[30]))
                bpm125PH.append(float(row[31]))
                bpm125PV.append(float(row[32]))


    plt.plot(v101, bpm101PH, '-o', label='B101PH')
    plt.plot(v101, bpm102PH, '-o', label='B102PH')
    plt.plot(v101, bpm103PH, '-o', label='B103PH')
    plt.plot(v101, bpm104PH, '-o', label='B104PH')
    plt.plot(v101, bpm106PH, '-o', label='B106PH')
    plt.plot(v101, bpm107PH, '-o', label='B107PH')
    plt.xlabel('H101 Current (A)')
    plt.ylabel('Beam Position (um)')
    plt.title('SLAC Chassis #2, 250 pC, 50 b, 0 Ampl, 300 shots')  # Hardcoded title
    plt.legend()
    plt.grid(True)
    plt.show()

    cc1_pos = 2.79763  # meters
    cc2_pos = 5.506526  # meters
    ccLen = 1.0  # meters
    bpmPos = [923.561,1432.107,4107.671,6819.571,8331.05,
              9142.588,10646.421,11534.404,13109.365,15724.128,
              16481.008,17341.977,18153.57075,19000,20000, 21000] # mm

    plt.plot(np.ones(5)*bpmPos[0]/1000, bpm101PH,'o', label='B101PH')
    plt.plot(np.ones(5)*bpmPos[1]/1000, bpm102PH,'o', label='B102PH')
    plt.plot(np.ones(5)*bpmPos[2]/1000, bpm103PH,'o', label='B103PH')
    plt.plot(np.ones(5)*bpmPos[3]/1000, bpm104PH,'o', label='B104PH')
    plt.plot(np.ones(5)*bpmPos[4]/1000, bpm106PH,'o', label='B106PH')
    plt.plot(np.ones(5)*bpmPos[5]/1000, bpm107PH,'o', label='B107PH')
    plt.plot(np.ones(5)*bpmPos[6]/1000, bpm111PH,'o', label='B111PH')
    plt.plot(np.ones(5)*bpmPos[7]/1000, bpm113PH,'o', label='B113PH')
    plt.plot(np.ones(5)*bpmPos[8]/1000, bpm117PH,'o', label='B117PH')
    plt.plot(np.ones(5)*bpmPos[9]/1000, bpm118PH,'o', label='B118PH')
    plt.plot(np.ones(5)*bpmPos[10]/1000, bpm120PH,'o', label='B120PH')
    plt.plot(np.ones(5)*bpmPos[11]/1000, bpm121PH,'o', label='B121PH')
    plt.plot(np.ones(5)*bpmPos[12]/1000, bpm122PH,'o', label='B122PH')
    plt.plot(np.ones(5)*bpmPos[13]/1000, bpm123PH,'o', label='B123PH')
    plt.plot(np.ones(5)*bpmPos[14]/1000, bpm124PH,'o', label='B124PH')
    plt.plot(np.ones(5)*bpmPos[15]/1000, bpm125PH,'o', label='B125PH')
    plt.xlabel('Beamline Position (um)')
    plt.ylabel('Beam Position (um)')
    plt.xlim([0, 22])
    plt.ylim([-15000, 15000])
    plt.vlines(cc1_pos - (ccLen/2.0), -10000, 10000)
    plt.vlines(cc1_pos + (ccLen/2.0), -10000, 10000)
    plt.vlines(cc2_pos - (ccLen/2.0), -10000, 10000)
    plt.vlines(cc2_pos + (ccLen/2.0), -10000, 10000)
    plt.text(cc1_pos , -9000, 'CC1', ha='center')
    plt.text(cc2_pos, -9000, 'CC2', ha='center')
    plt.title('SLAC Chassis #2, 250 pC, 50 b, 0 Ampl, 300 shots')  # Hardcoded title
    plt.legend()
    plt.show()


def charge_CMHOM(data_file):
    with open(data_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        charge = []
        bunches = []
        shots =[]
        c1 = []
        c2 = []
        c3 = []
        c4 = []
        c5 = []
        c6 = []
        c7 = []
        c8 = []
        c1_175 = []
        c8_175 = []
        c1_250 = []
        c8_250 = []
        c1_325 = []
        c8_325 = []
        for row in readCSV:
            if 'time' not in row:
                bunches.append(float(row[1]))
                charge.append(float(row[2]))
                shots.append(float(row[3]))
                c1.append(float(row[4]))
                c2.append(float(row[5]))
                c3.append(float(row[6]))
                c4.append(float(row[7]))
                c5.append(float(row[8]))
                c6.append(float(row[9]))
                c7.append(float(row[10]))
                c8.append(float(row[11]))
                c1_175.append(float(row[12]))
                c8_175.append(float(row[13]))
                c1_250.append(float(row[14]))
                c8_250.append(float(row[15]))
                c1_325.append(float(row[16]))
                c8_325.append(float(row[17]))


    data = {'bunches' : bunches, 'charge' : charge, 'shots' : shots,
            'c1' : c1, 'c2' : c2, 'c3' : c3, 'c4' : c4,
            'c5' : c5, 'c6' : c6, 'c7' : c7, 'c8' : c8,
            'c1_175' : c1_175, 'c8_175' : c8_175,'c1_250' : c1_250,
            'c8_250' : c8_250, 'c1_325' : c1_325,'c8_325' : c8_325,}

    df = pd.DataFrame(data,columns=['bunches','charge','shots',
            'c1','c2','c3','c4','c5','c6','c7','c8',
            'c1_175','c8_175','c1_250',
            'c8_250','c1_325','c8_325'])

    #df.plot.scatter(x='charge', y='c1', s=df['bunches'])
    sns.catplot(x="charge", y="c1", col='bunches', data=df,label='c1')
    sns.scatterplot(x="charge", y="c2", data=df,label='c2')
    plt.show()
    exit()

    plt.plot(charge, c1, '*', label='C1')
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.scatter(charge[0:2],c1[0:2], c=color[0], marker='o', label='C1')
    plt.scatter(charge[2:4],c1[2:4], c=color[0], marker='*')
    plt.scatter(charge[4:5],c1[4:5], c=color[0], marker='^')
    plt.scatter(charge[0:2],c2[0:2], c=color[1], marker='o', label='C2')
    plt.scatter(charge[2:4],c2[2:4], c=color[1], marker='*')
    plt.scatter(charge[4:5],c2[4:5], c=color[1], marker='^')
    plt.scatter(charge[0:2],c3[0:2], c='r', marker='o', label='C3')
    plt.scatter(charge[2:4],c3[2:4], c='r', marker='*')
    plt.scatter(charge[4:5],c3[4:5], c='r', marker='^')
    plt.scatter(charge[0:2],c4[0:2], c='c', marker='o', label='C4')
    plt.scatter(charge[2:4],c4[2:4], c='c', marker='*')
    plt.scatter(charge[4:5],c4[4:5], c='c', marker='^')
    plt.scatter(charge[0:2],c5[0:2], c='m', marker='o', label='C5')
    plt.scatter(charge[2:4],c5[2:4], c='m', marker='*')
    plt.scatter(charge[4:5],c5[4:5], c='m', marker='^')
    plt.scatter(charge[0:2],c6[0:2], c='y', marker='o', label='C6')
    plt.scatter(charge[2:4],c6[2:4], c='y', marker='*')
    plt.scatter(charge[4:5],c6[4:5], c='y', marker='^')
    plt.scatter(charge[0:2],c7[0:2], c='k', marker='o', label='C7')
    plt.scatter(charge[2:4],c7[2:4], c='k', marker='*')
    plt.scatter(charge[4:5],c7[4:5], c='k', marker='^')
    plt.scatter(charge[0:2],c8[0:2], c='w', marker='o', label='C8')
    plt.scatter(charge[2:4],c8[2:4], c='w', marker='*')
    plt.scatter(charge[4:5],c8[4:5], c='w', marker='^')
    plt.xlabel('Bunch Charge (pC)')
    plt.ylabel('HOM Signal Peak (V)')
    plt.title('SLAC Chassis #2, 100 pC, 1 b, 1 Ampl, 100 shots')  # Hardcoded title
    plt.legend()
    plt.grid(False)
    plt.show()


def charge_CMHOM_pd(data_file):
    with open(data_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        charge = []
        bunches = []
        shots =[]
        cav = []
        hom = []
        loc = []
        
        for row in readCSV:
            if 'time' not in row:
                bunches.append(int(row[1]))
                charge.append(int(row[2]))
                shots.append(int(row[3]))
                cav.append(int(row[4]))
                hom.append(float(row[5]))
                loc.append(row[6])

    data = {'bunches' : bunches, 'charge' : charge, 'shots' : shots,
            'cav' : cav, 'hom' : hom, 'loc' : loc}

    df = pd.DataFrame(data,columns=['bunches','charge','shots',
            'cav', 'hom', 'loc'])

    slac_cavs = [1,2,3,4,5,6,7,8]
    df_slac = df[df.cav.isin(slac_cavs)]
    g = sns.catplot(x="charge", y="hom", hue='cav', row='bunches', col='loc', jitter=False, data=df_slac, sharey=False)
    g.set_xlabels('Bunch Charge (pC)')
    g.set_ylabels('HOM Peak (V)')
    #plt.show(g)

    # Plot suggested by Alex
    alex_plt = sns.catplot(x="cav", y="hom", hue='charge', col='loc', row='bunches', jitter=False, data=df_slac, sharey=False)
    alex_plt.set_xlabels('Cavity')
    alex_plt.set_ylabels('HOM Peak (V)')
    #title = ('Up and downstream HOM peaks V125 at -1 Amp\n' +
    #         'SLAC Chassis, 250 pC, 50 b, 1 Ampl, 300 shots')  # Hardcoded title
    #plt.title(title)
    plt.show(alex_plt)

if __name__ == "__main__":
    des = 'Procces and plot FAST data'
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('-f', '--file', dest='data_file', required=True,
                        help='File with data to be processed/plotted')
    parser.add_argument('-hom', action="store_true",
                        help='Plot Capture Cavities (CC) HOM raw data')
    parser.add_argument('-cmhom', action="store_true",
                        help='Plot CMHOM raw data')
    parser.add_argument('-bpmmag', action="store_true",
                        help='Plot BPM magnets raw data')
    parser.add_argument('-bpm', action="store_true",
                        help='Plot BPM raw data')
    parser.add_argument('-lebpm', action="store_true",
                        help='Plot LEBPM raw data')
    parser.add_argument('-hebpm', action="store_true",
                        help='Plot HEBPM raw data')
    parser.add_argument('-tor', action="store_true",
                        help='Plot toroid raw data')
    parser.add_argument('-phom', action="store_true",
                        help='Process HOM data: Remove baseline and calculate peak mean and std')
    parser.add_argument('-pcmhom', action="store_true",
                        help='Process CMHOM data: Remove baseline and calculate peak mean and std')
    parser.add_argument('-pbpm', action="store_true",
                        help='Process BPM data: Plot Beamline Position Vs BPM average reading')
    parser.add_argument('-plebpm', action="store_true",
                        help='Process LEBPM data: Plot Beamline Position Vs BPM average reading')
    parser.add_argument('-vh', action="store_true",
                        help='Plot V/H101 vs HOMs')
    parser.add_argument('-vb', action="store_true",
                        help='Plot V/H101 vs BPMs')
    parser.add_argument('-ccm', action="store_true",
                        help='Plot charge vs CMHOM')
    parser.add_argument('-cccmhom', action="store_true",
                        help='Plot corrector current vs CMHOM')
    parser.add_argument('-intcmhom', action="store_true",
                        help='Calculate integrals of CM HOM signals')
    parser.add_argument('-s', action="store_true",
                        help='Save plot')
    parser.add_argument('-m', metavar='mode', dest='mode', type=int,
                        choices=range(0, 100), default=0,
                        help='Select the device number. 0 to plot all devices')

    args = parser.parse_args()

    if args.vh:
        V101_HOM(args.data_file)
        exit()
    if args.vb:
        V101_BPM(args.data_file)
        exit()
    if args.ccm:
        charge_CMHOM_pd(args.data_file)
        exit()
    if args.cccmhom:
        CMHOM_125(args.data_file)
        exit()

    data = sio.loadmat(args.data_file)
    HOMsList = ['CC1 Upstream', 'CC1 Downstream', 'CC2 Upstream',
                'CC2 Downstream', 'C1IW1A?', 'C1IW1B?']
    data['HOMs_List'] = HOMsList
    CMHOMsList = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
                  'C1_1.75', 'C8_1.75', 'C1_2.5', 'C8_2.5',
                  'C1_3.25', 'C8_3.25',]
    data['CMHOMs_List'] = CMHOMsList
    bpmPos = [0.923561,1.432107,4.107671,6.819571,8.33105,9.142588]  # meters
    data['bpmPos'] = bpmPos
    BPMsList = ['B101PH', 'B101PV', 'B102PH', 'B102PV',
                'B103PH', 'B103PV', 'B104PH', 'B104PV',
                'B106PH', 'B106PV', 'B107PH', 'B107PV',
                'B111PH', 'B111PV', 'B113PH', 'B113PV',
                'B117PH', 'B117PV', 'B118PH', 'B118PV',
                'B120PH', 'B120PV', 'B121PH', 'B121PV',
                'B122PH', 'B122PV', 'B123PH', 'B123PV',
                'B124PH', 'B124PV', 'B125PH', 'B125PV',
                ]
    data['BPMsList'] = BPMsList
    LEBPMsList = ['B101PH', 'B101PV', 'B102PH', 'B102PV',
                'B103PH', 'B103PV', 'B104PH', 'B104PV',
                'B106PH', 'B106PV', 'B107PH', 'B107PV',
                'B111PH', 'B111PV', 'B113PH', 'B113PV',
                'B117PH', 'B117PV', 'B118PH', 'B118PV',
                'B120PH', 'B120PV', 'B121PH', 'B121PV',
                'B122PH', 'B122PV', 'B123PH', 'B123PV',
                'B124PH', 'B124PV', 'B125PH', 'B125PV',
                'B130PH', 'B130PV']
    data['LEBPMsList'] = LEBPMsList
    
    basic_file_info(data)
    if args.phom:
        process_HOM_data(data)
    if args.pbpm:
        process_BPM_data(data, "BPMs")
    if args.plebpm:
        process_BPM_data(data, "LEBPMs")
    if args.tor:
        plot_toroids(data, args.mode)
    if args.bpmmag:
        plot_BPMMags(data, args.mode)
    if args.bpm:
        plot_BPM(data, args.mode, 'BPMs')
    if args.hebpm:
        plot_BPM(data, args.mode, 'HEBPMs')
    if args.lebpm:
        plot_BPM(data, args.mode, 'LEBPMs')
    if args.hom:
        plot_raw_HOMs(data, args.mode)
    if args.cmhom:
        plot_raw_CMHOMs(data, args.mode, args.s)
    if args.pcmhom:
        process_CMHOM_data(data, args.s)
    if args.intcmhom:
        integral_CMHOM(data, args.s)
        exit()
