import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.io as sio
import csv
import pandas as pd
import seaborn as sns
import pickle 


def process_BPM_var(data):
    print("Processing BPM data...")
    BPMs = data['HEBPMs']
    num_BPMs = BPMs.shape[0]
    reps = BPMs.shape[1]
    points = BPMs.shape[2] - 6
    print('Num BPMs = ', num_BPMs)
    print('BPMs Repetitions = ', reps)
    print('BPMs Points = ', points)

    bpm_means = np.mean(BPMs, axis=1) # Mean over the number of reps(shots)
    bpm_means = np.delete(bpm_means, (0,1,2,3,-2,-1), 1) # Remove weird points at start and finish of array
    bpm_mean_means = np.mean(bpm_means, axis=1) # Means over the number of bunches
    bpm_vars = np.var(bpm_means, axis=1) # variance over the number of bunches

    return bpm_means, bpm_mean_means, bpm_vars
     

def plot_BPM_data(data, bpm_means, bpm_mean_means, bpm_vars):
    num_BPMs = 8 # I am interested in the vertical reading of the first 4: B418, 440, 441, 444 ... for now
    num_rows = 2
    num_cols = 4
    fig, axs = plt.subplots(num_rows, num_cols, sharey=True, sharex=True)
    for bpm in range(0,num_BPMs):
        bpm_name = data['HEBPMsList'][bpm]
        bpm_mean = bpm_means[bpm]
        bpm_mean_mean = bpm_mean_means[bpm]
        bpm_var = bpm_vars[bpm]

        axs[int(bpm / num_cols), bpm % num_cols].plot(bpm_mean - bpm_mean_mean)
        axs[int(bpm / num_cols), bpm % num_cols].grid(True)
        title = (bpm_name + " var = " + "{:.2f}".format(bpm_var))
        axs[int(bpm / num_cols), bpm % num_cols].set_title(title)
    plt.show()


def create_big_dataset(data, load):
    df_file = 'HOM_FAST_20201106/bpm_big_data_shift6.plk'

    columns=['connection','bunch_charge','num_bunches','V125',
             'H125','reps','BPM_attenuator','SLAC_attenuator','data_file']

    col_data_LE = [s + '_data' for s in data['LEBPMsList']]
    col_mean_LE = [s + '_mean' for s in data['LEBPMsList']]
    col_var_LE = [s + '_var' for s in data['LEBPMsList']]
    col_std_LE = [s + '_std' for s in data['LEBPMsList']]
    col_data = [s + '_data' for s in data['HEBPMsList']]
    col_mean = [s + '_mean' for s in data['HEBPMsList']]
    col_var = [s + '_var' for s in data['HEBPMsList']]
    col_std = [s + '_std' for s in data['HEBPMsList']]
    columns = columns + data['CMHOMs_List'] + col_data_LE + col_mean_LE + col_var_LE + col_std_LE + col_data + col_mean + col_var + col_std

    if load == True:
        df = pd.read_pickle(df_file) 
    else:
        df = pd.DataFrame(columns=columns)

    connection =  data['connection']
    bunch_charge = data['bunch_charge']
    num_bunches =  data['num_bunches']
    V125 =  data['V125']
    H125 =  data['H125']
    reps =  data['reps']
    BPM_attenuator =  data['BPM_attenuator']
    SLAC_attenuator =  data['SLAC_attenuator']
    data_file =  data['data_file']
    
    LEBPMs = data['LEBPMs']
    num_LEBPMs = LEBPMs.shape[0]

    HEBPMs = data['HEBPMs']
    num_HEBPMs = HEBPMs.shape[0]

    reps = HEBPMs.shape[1]
    points = HEBPMs.shape[2] - 6

    CMHOMs = data['CMHOMs']
    num_CMHOMs = CMHOMs.shape[0]

    for rep in range(0, reps):
        row = [connection, bunch_charge, num_bunches,
        V125, H125, reps, BPM_attenuator, SLAC_attenuator, data_file]
        for cmhom in range(0, num_CMHOMs):
            cmhom_baseln_a = CMHOMs[cmhom][rep][100:200]
            cmhom_baseln_b = CMHOMs[cmhom][rep][1000:1500]
            cmhom_baseln = np.mean(np.concatenate((cmhom_baseln_a, cmhom_baseln_b)))
            cmhom_based = CMHOMs[cmhom][rep] - cmhom_baseln
            if cmhom < 12:
                cmhom_pk_bsd = np.max(cmhom_based)
            elif cmhom >=12:
                cmhom_pk_bsd = np.min(cmhom_based)
            row.extend([cmhom_pk_bsd])

        lebpms_data = []
        lebpms_mean = []
        lebpms_var = []
        lebpms_std = []
        for lebpm in range(0, num_LEBPMs):
            lebpm_data = LEBPMs[lebpm][rep]
            lebpm_data = np.delete(lebpm_data, (0,1,2,3,-2,-1)) # Remove weird points at start and finish of array
            lebpm_mean = np.mean(lebpm_data)
            lebpm_var = np.var(lebpm_data)
            lebpm_std = np.std(lebpm_data)
            lebpms_data.append(lebpm_data - lebpm_mean)
            lebpms_mean.append(lebpm_mean)
            lebpms_var.append(lebpm_var)
            lebpms_std.append(lebpm_std)
        row.extend(lebpms_data)
        row.extend(lebpms_mean)
        row.extend(lebpms_var)
        row.extend(lebpms_std)

        hebpms_data = []
        hebpms_mean = []
        hebpms_var = []
        hebpms_std = []
        for hebpm in range(0, num_HEBPMs):
            hebpm_data = HEBPMs[hebpm][rep]
            hebpm_data = np.delete(hebpm_data, (0,1,2,3,-2,-1)) # Remove weird points at start and finish of array
            hebpm_mean = np.mean(hebpm_data)
            hebpm_var = np.var(hebpm_data)
            hebpm_std = np.std(hebpm_data)
            hebpms_data.append(hebpm_data - hebpm_mean)
            hebpms_mean.append(hebpm_mean)
            hebpms_var.append(hebpm_var)
            hebpms_std.append(hebpm_std)
        row.extend(hebpms_data)
        row.extend(hebpms_mean)
        row.extend(hebpms_var)
        row.extend(hebpms_std)

        df_tmp = pd.DataFrame([row], columns=columns)
        df = df.append(df_tmp, ignore_index=True)
    
    df.to_pickle(df_file)


def create_dataset(data, bpm_means, bpm_mean_means, bpm_vars, cmhoms_pk_bsd_mean, cmhoms_pk_bsd_std, load):
    columns=['bunch_charge', 'num_bunches', 'V125', 
             'H125', 'connection', 'reps']
    col_mean = [s + '_mean' for s in data['HEBPMsList']]
    col_var = [s + '_var' for s in data['HEBPMsList']]
    col_data = [s + '_data' for s in data['HEBPMsList']]
    columns = columns + data['CMHOMs_List'] + col_data + col_mean + col_var

    bunch_charge = data['bunch_charge']
    num_bunches =  data['num_bunches']
    V125 =  data['V125']
    H125 =  data['H125']
    connection =  data['connection']
    
    BPMs = data['HEBPMs']
    reps = BPMs.shape[1]

    row = [bunch_charge, num_bunches,
           V125, H125, connection, reps]
    row.extend(cmhoms_pk_bsd_mean)
    for bpm in range(0,len(bpm_vars)):
        row.extend([bpm_means[bpm]])

    row.extend(bpm_mean_means)
    row.extend(bpm_vars)

    if load == True:
        df = pd.read_pickle('HOM_FAST_20201106/bpm_data_ex.plk') 
        df_tmp = pd.DataFrame([row], columns=columns)
        df = df.append(df_tmp, ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=columns)
        
    df.to_pickle('HOM_FAST_20201106/bpm_data_ex.plk') # where to save it usually as a .plk
    print(df)
    

def process_CMHOM_peaks(data):
    CMHOMs = data['CMHOMs']
    num_CMHOMs = CMHOMs.shape[0]
    reps = CMHOMs.shape[1]

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
        cmhoms_pk_bsd_mean[cmhom] = np.mean(cmhoms_pks_bsd)
        cmhoms_pk_bsd_std[cmhom] = np.std(cmhoms_pks_bsd)

    print('cmhoms_peak_based_mean = ', cmhoms_pk_bsd_mean)
    print('cmhoms_peak_based_std = ', cmhoms_pk_bsd_std)

    return cmhoms_pk_bsd_mean, cmhoms_pk_bsd_std


if __name__ == "__main__":
    des = 'Procces FAST data for ML purposes'
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('-f', '--file', dest='data_file', required=False,
                        help='File with data to be processed/plotted')

    args = parser.parse_args()

    # Read list of datafiles to process
    datafiles = pd.read_csv('HOM_FAST_20201106/list_of_files_shift6.csv')
    for datafile in range(0, len(datafiles.index)):
        data_file = datafiles['data_file'][datafile]
        data = sio.loadmat('HOM_FAST_20201106/' + data_file)
        print('processing file #' + str(datafile) + ": " + data_file)

        data['connection'] = datafiles['connection'][datafile]
        data['bunch_charge'] = datafiles['bunch_charge'][datafile]
        data['num_bunches'] = datafiles['num_bunches'][datafile]
        data['V125'] = datafiles['V125'][datafile]
        data['H125'] = datafiles['H125'][datafile]
        data['reps'] = datafiles['reps'][datafile]
        data['BPM_attenuator'] = datafiles['BPM_attenuator'][datafile]
        data['SLAC_attenuator'] = datafiles['SLAC_attenuator'][datafile]
        data['data_file'] = datafiles['data_file'][datafile]

        LEBPMsList = ['B101PH', 'B101PV', 'B102PH', 'B102PV',
                  'B103PH', 'B103PV', 'B104PH','B104PV',
                  'B106PH', 'B106PV', 'B107PH', 'B107PV',
                  'B111PH', 'B111PV', 'B113PH', 'B113PV',
                  'B117PH', 'B117PV', 'B118PH', 'B118PV',
                  'B120PH', 'B120PV', 'B121PH', 'B121PV',
                  'B122PH', 'B122PV', 'B123PH', 'B123PV',
                  'B124PH', 'B124PV', 'B125PH', 'B125PV',
                  'B130PH', 'B130PV']
        HEBPMsList = ['B418PH', 'B418PV', 'B440PH', 'B440PV',
                  'B441PH', 'B441PV', 'B444PH', 'B444PV',
                  'B450PH', 'B450PV', 'B455PH', 'B455PV',
                  'B460PH', 'B460PV', 'B470PH', 'B470PV',
                  'B480PH', 'B480PV', 'B501PH', 'B501PV',
                  'B505PH', 'B505PV', 'B506PH', 'B506PV',
                  'B507PH', 'B507PV', 'B508PH', 'B508PV',
                  'B512PH', 'B512PV', 'B513PH', 'B513PV',
                  'B514PH', 'B514PV', 'B600PH', 'B600PV',
                  'B601PH', 'B601PV', 'B603PH', 'B603PV',
                  'B604PH', 'B604PV', 'B605PH', 'B605PV',
                  'B609PH', 'B609PV', 'B610PH', 'B610PV',
                  'B612PH', 'B612PV', 'B613PH', 'B613PV']
        data['LEBPMsList'] = LEBPMsList
        data['HEBPMsList'] = HEBPMsList

        CMHOMsList = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
                  'C1_1.75', 'C8_1.75', 'C1_2.5', 'C8_2.5',
                  'C1_3.25', 'C8_3.25',]
        data['CMHOMs_List'] = CMHOMsList


        if datafile == 0:
            load=False
        else:
            load=True
        create_big_dataset(data, load=load)


    #cmhoms_pk_bsd_mean, cmhoms_pk_bsd_std = process_CMHOM_peaks(data)
    #bpm_means, bpm_mean_means, bpm_vars = process_BPM_var(data)
    #plot_BPM_data(data, bpm_means, bpm_mean_means, bpm_vars)
    #create_dataset(data, bpm_means, bpm_mean_means, bpm_vars, cmhoms_pk_bsd_mean, cmhoms_pk_bsd_std, load=True)