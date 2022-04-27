import numpy as np
from os import walk
import argparse
import json
import csv
import datetime


def sim_directories(path):
    sim_dirs = []
    for (dirpath, dirnames, filenames) in walk('.'):
        for directory in dirnames:
            if 'simulation' in directory: sim_dirs.append(directory)
        break
    
    return sim_dirs


def process_sim_data(data_file, data_dir, small):
    print("Loading data from ", data_dir)

    # Load simulation data
    with open(data_dir + '/data.npy', 'rb') as f:
        t = np.load(f)
        cav_v_s = np.load(f)
        fwd_s = np.load(f)

    
    # Load Simulation settings
    with open(data_dir + '/simulation_settings.json') as json_file:
        settings = json.load(json_file)
    stable_gbws = settings['stable_gbws']
    tf = settings['tf']
    steady_state_start = settings["steady_state_start"]
    fund_k_drive =  settings["fund_k_drive"]
    ssa_power  = settings["ssa_power"]
    amp_stability = settings["amp_stability"]
    phase_stability = settings["phase_stability"]
    ssa_limit = settings["ssa_limit"]
    tf = settings["tf"]
    nom_grad = settings["nom_grad"]
    beam = settings["beam"]
    detuning = settings["detuning"]
    noise = settings["noise"]
    feedforward = settings["feedforward"]
    noise_psd = settings["noise_psd"]
    beam_start = settings["beam_start"]
    beam_end = settings["beam_end"]
    detuning_start = settings["detuning_start"]
    detuning_end = settings["detuning_end"]
    stable_gbws = settings["stable_gbws"]
    control_zero = settings["control_zero"]

    bw = 104

    with open(data_file, 'a', newline='') as f:
        thewriter = csv.writer(f)
        for count, stable_gbw in enumerate(stable_gbws):
            print('Processing 0db crossing ' + str(stable_gbw))
            p_gain = stable_gbw*2*np.pi/bw

            point_number = 20000 if small else len(cav_v_s[count])

            for point in range(0,point_number):

                if point == 0:
                    sim_start = 1
                else:
                    sim_start = 0

                if beam and (t[point] >= beam_start) and (t[point] <= beam_end):
                    beam_current = 100.0 * 10.0 ** (-6)
                else:
                    beam_current = 0

                if detuning and (t[point] >= detuning_start) and (t[point] <= detuning_end):
                    detuning_amp = 10
                    detuning_freq = 100
                else:
                    detuning_amp = 0
                    detuning_freq = 0

                if noise:
                    meas_psd = noise_psd
                else:
                    meas_psd = 0

                if feedforward == 'True':
                    ff = 1
                else:
                    ff = 0
            
                thewriter.writerow([sim_start, t[point], np.real(cav_v_s[count][point]), np.imag(cav_v_s[count][point]),
                                   np.real(fwd_s[count][point]), np.imag(fwd_s[count][point]), 
                                   stable_gbw, p_gain, nom_grad, beam_current, detuning_amp, detuning_freq,
                                   meas_psd, ff])




if __name__ == "__main__":
    des = 'Collect all the data from simulations to build a dataset'
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('-s', "--small", action="store_true",
                        help='Build "small" dataset (~1M points)')

    args = parser.parse_args()

    # Create list of simulation directories 
    sim_dirs = sim_directories('.')

    # Create datset file 
    creation_time = datetime.datetime.now()
    data_file = creation_time.strftime('dataset_' + '%Y%m%d_%H%M%S' + '.csv')
    with open(data_file, 'w', newline='') as f:
        thewriter = csv.writer(f)
        header = ['sim_start','time','cav_real','cav_imag','fwd_real','fwd_imag','0db_crossing','gain','sp','beam_current','detuning_amp',
                  'detuning_freq','meas_psd','feedforward']
        thewriter.writerow(header)

    # Load and write data to dataset
    dir_number = 4 if args.small else len(sim_dirs) 
    print('The script will process ' + str(dir_number) + ' directories')
    for i in range(0,dir_number):
        process_sim_data(data_file, sim_dirs[i], args.small)
