import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
import datetime
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from cavity_step_response import cavity_step as cav_stp

plt.rcParams.update({'font.size': 12})
global steady_state_start
steady_state_start = 12500  # microseconds


def error_integral(time, sp, vol):
    e_int = [0]
    for t in range(len(time)):
        if t > steady_state_start:
            e = np.abs(sp - vol[t])
            temp_int = e_int[-1] + e
            e_int.append(temp_int)
    return e_int


def test_error_integral():
    t = np.linspace(0, 4*np.pi, 25000)
    sp = 0.0
    vol = np.sin(t)
    ei = error_integral(t, sp, vol)
    plt.plot(t[steady_state_start:], np.ones(len(t[steady_state_start:])) * sp,
             '--', label='Set Point')
    plt.plot(t[steady_state_start:], vol[steady_state_start:], label='Signal')
    plt.plot(t[steady_state_start:], ei[0:len(t[steady_state_start:])],
             label='Error integral')
    plt.show()


def error(sp, vol):
    error = sp - vol
    return error


def rmse(error):
    n = len(error[steady_state_start:-1])
    mse = np.sum(np.abs(error[steady_state_start:-1])**2) / n
    rmse = np.sqrt(mse)
    return rmse


def log(data_dir, settings):
    with open(data_dir + "/simulation_settings.json", "w") as f:
        json.dump(settings, f)
    print("Simulation settings saved in ", data_dir,
          "/simulation_settings.json")
    return 0


def max_error(sp, vol):
    upper_error = np.amax(np.abs(vol[steady_state_start:-1])) - sp
    lower_error = sp - np.amin(np.abs(vol[steady_state_start:-1]))
    max_error = np.amax([abs(upper_error), abs(lower_error)])
    return max_error


if __name__ == "__main__":
    des = 'Optimize proportional and integral gains of the LLRF\
           controller by minimizing the integral of the cav_v error'
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('-ei', "--e_integral", action="store_true",
                        help='test error integral function')
    parser.add_argument('-p', "--path", dest="path", default=None,
                        help='Path to plot data generated previously')

    args = parser.parse_args()

    # Test integral of error funcion
    if args.e_integral:
        test_error_integral()
        exit()

    # Some useful constants
    fund_k_drive = 407136.3407999831
    ssa_power = 3800  # W

    # LLRF requirements
    amp_stability = 0.0001  # 0.01%
    phase_stability = 0.01  # degress
    ssa_limit = 0.04 * fund_k_drive * np.sqrt(ssa_power)

    # Some settings for the simulation
    tf = 0.03  # Total simulation time
    nom_grad = 1.6301e7  # V/m. Nominal gradient of LCLS-II cavity
    beam = False
    detuning = True
    noise = True
    feedforward = True
    noise_psd = -138
    beam_start = 0.015
    beam_end = 0.04
    detuning_start = 0.0
    detuning_end = tf

    # Gain Parameters
    # stable_gbws=[70000, 60000, 50000, 40000, 30000, 20000, 10000, 5000]
    # control_zero=[17500, 15000, 12500, 10000, 7500, 5000, 2500, 1750]
    # stable_gbws=[50000, 40000, 20000]
    # control_zero=[12500, 10000, 5000]
    stable_gbws = [70000]
    control_zero = [17500]
    # stable_gbws=[40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000,
    # 40000, 40000]
    # control_zero=[10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000,
    # 10000, 10000]

    if not args.path:

        # Save log file
        start_time = datetime.datetime.now()
        data_dir = start_time.strftime('simulation_' + '%Y%m%d_%H%M%S')
        os.mkdir(data_dir)
        print("Creating directory ", data_dir)
        simulation_settings = {
            "steady_state_start": steady_state_start,
            "fund_k_drive": fund_k_drive,
            "ssa_power": ssa_power,
            "amp_stability": amp_stability,
            "phase_stability": phase_stability,
            "ssa_limit": ssa_limit,
            "tf": tf,
            "nom_grad": nom_grad,
            "beam": beam,
            "detuning": detuning,
            "noise": noise,
            "feedforward": feedforward,
            "noise_psd": noise_psd,
            "beam_start": beam_start,
            "beam_end": beam_end,
            "detuning_start": detuning_start,
            "detuning_end": detuning_end,
            "stable_gbws": stable_gbws,
            "control_zero": control_zero
        }
        log(data_dir, simulation_settings)

        args = [tf, nom_grad, feedforward, noise, noise_psd, beam, beam_start,
                beam_end, detuning, detuning_start, detuning_end]

        # Arrays for the for loop
        cav_v_s = []
        fwd_s = []
        e_s = []
        ie_s = []
        kp_s = []
        ki_s = []
        for s_gbw in stable_gbws:
            print("Generating data for K_p = ", s_gbw)
            t, cav_v, sp, fwd, Kp_a, Ki_a, e, ie = cav_stp(args,
                                                           stable_gbw=s_gbw)
            ei = error_integral(t, sp, cav_v)
            cav_v_s.append(cav_v)
            fwd_s.append(fwd[:, 0])
            e_s.append(e)
            ie_s.append(ei)
            kp_s.append(Kp_a)
            ki_s.append(Ki_a)

        # Calculate maximum error and integral of error
        final_ie = []
        maximum_error = []
        errors = []
        for gain in range(len(stable_gbws)):
            errors.append(error(sp, np.abs(cav_v_s[gain])))
            final_ie.append(ie_s[gain][-1])
            maximum_error.append(max_error(sp, cav_v_s[gain]))
            print("maximum error for", stable_gbws[gain],  "is ",
                  maximum_error[gain])

        # Save waveform results
        with open(data_dir + '/data.npy', 'wb') as f:
            print("Saving data in", data_dir, "/data.npy")
            np.save(f, t)
            np.save(f, cav_v_s)
            np.save(f, fwd_s)
            np.save(f, errors)
            np.save(f, ie_s)
            np.save(f, final_ie)
            np.save(f, maximum_error)

    # Open Results
    if args.path:
        sp = 16301000.0
        data_dir = args.path
        print("Loading data from ", data_dir)
        with open(data_dir + '/data.npy', 'rb') as f:
            t = np.load(f)
            cav_v_s = np.load(f)
            fwd_s = np.load(f)
            errors = np.load(f)
            ie_s = np.load(f)
            final_ie = np.load(f)
            maximum_error = np.load(f)

        print("Loading simulation settings from ", data_dir)
        f = open(data_dir + '/simulation_settings.json')
        simulation_settings = json.load(f)

        steady_state_start = simulation_settings['steady_state_start']
        fund_k_drive = simulation_settings['fund_k_drive']
        ssa_power = simulation_settings['ssa_power']
        amp_stability = simulation_settings['amp_stability']
        phase_stability = simulation_settings['phase_stability']
        ssa_limit = simulation_settings['ssa_limit']
        tf = simulation_settings['tf']
        nom_grad = simulation_settings['nom_grad']
        beam = simulation_settings['beam']
        detuning = simulation_settings['detuning']
        noise = simulation_settings['noise']
        feedforward = simulation_settings['feedforward']
        noise_psd = simulation_settings['noise_psd']
        beam_start = simulation_settings['beam_start']
        beam_end = simulation_settings['beam_end']
        detuning_start = simulation_settings['detuning_start']
        detuning_end = simulation_settings['detuning_end']
        stable_gbws = simulation_settings['stable_gbws']
        control_zero = simulation_settings['control_zero']
        f.close()

    # Calculate rmse and std
    rmse_s = []
    rmse_sk = []
    std_s = []
    for gain in range(len(stable_gbws)):
        rmse_s.append(rmse(errors[gain]))
        std_s.append(np.std(errors[gain][steady_state_start:-1]))
        from sklearn.metrics import mean_squared_error
        length = len(cav_v_s[gain][steady_state_start:-1])
        mse_sk = mean_squared_error(np.ones(length)*sp,
                                    np.abs(cav_v_s[gain][steady_state_start:-1]))
        rmse_sk.append(np.sqrt(mse_sk))

    fig, axs = plt.subplots(2, 3)
    # some plot settings
    t = t * 1000  # seconds to miliseconds
    x_lim = [steady_state_start / 1000, tf * 1000]  # Time window in msec
    y_lim = [nom_grad-6e3, nom_grad+6e3]
    for p in range(len(stable_gbws)):
        axs[0, 0].set_title('Cavity Probe')
        axs[0, 0].plot(t, np.abs(cav_v_s[p]), label=stable_gbws[p])
        axs[0, 0].plot(t, sp * np.ones(len(t)), 'k--', linewidth=2)
        axs[0, 0].set_ylabel('Amplitude [V]')
        axs[0, 0].legend(loc='lower right')

        axs[1, 0].set_title('Cavity Probe Steady State')
        axs[1, 0].plot(t, np.abs(cav_v_s[p]),
                       label=str(round(stable_gbws[p], 3))
                       + "--" + str(round(rmse_s[p], 3))
                       + "--" + str(round(std_s[p], 3))
                       + "---" + str(round(rmse_sk[p], 3)))
        axs[1, 0].plot(t, nom_grad*np.ones(len(t))*(1 + amp_stability),
                       linewidth=2, color='m')
        axs[1, 0].plot(t, nom_grad*np.ones(len(t))*(1 - amp_stability),
                       linewidth=2, color='m')
        axs[1, 0].plot(t, sp * np.ones(len(t)), 'k--', linewidth=2)
        axs[1, 0].set_xlim(x_lim)
        axs[1, 0].set_ylim(y_lim)
        axs[1, 0].set_xlabel('Time [ms]')
        axs[1, 0].set_ylabel('Amplitude [V]')
        axs[1, 0].legend(loc='lower right')

        axs[0, 1].set_title('Forward/control signal')
        axs[0, 1].plot(t, np.abs(fwd_s[p]), label=stable_gbws[p])
        axs[0, 1].plot(t, sp * np.ones(len(t)), 'k--', linewidth=2)
        axs[0, 1].legend(loc='lower right')

        axs[1, 1].set_title('Forward/control signal Steady State')
        axs[1, 1].plot(t, np.abs(fwd_s[p]), label=stable_gbws[p])
        axs[1, 1].plot(t, sp * np.ones(len(t)), 'k--', linewidth=2)
        axs[1, 1].plot(t, np.ones(len(t)) * (sp + ssa_limit), linewidth=2,
                       color='m')
        axs[1, 1].plot(t, np.ones(len(t)) * (sp - ssa_limit), linewidth=2,
                       color='m')
        axs[1, 1].set_xlim(x_lim)
        axs[1, 1].set_ylim([nom_grad-25e5, nom_grad+25e5])
        axs[1, 1].set_xlabel('Time [ms]')
        axs[1, 1].legend(loc='lower right')

        # axs[0, 2].set_title('Forward/control power')
        axs[0, 2].set_title('Error')
        # axs[0, 2].plot(t, (np.abs(fwd_s[p])**2.0) / 1036,
        # label=stable_gbws[p])  # power
        axs[0, 2].plot(t, (np.abs(errors[p])), label=stable_gbws[p])  # Error
        # axs[0, 2].set_ylabel('Power [W]')
        axs[0, 2].legend(loc='lower right')

        axs[1, 2].set_title('Integral of the error Steady State')
        axs[1, 2].plot(t[steady_state_start-1:-1], np.abs(ie_s[p][0:len(t)]),
                       label=stable_gbws[p])
        axs[1, 2].set_xlabel('Time [ms]')
        axs[1, 2].legend(loc='upper left')

    plt.show()

    print("Maximum error = ", maximum_error)
    print("Final integral of the error = ", final_ie)
    print("Error RMS = ", rmse_s)
    print("RMSE STD = ", np.std(rmse_s))
    print("RMSE mean = ", np.mean(rmse_s))

    plt.figure()
    plt.title('Error Integral vs P_gain')
    print(stable_gbws)
    print(final_ie)
    plt.plot(stable_gbws, final_ie)
    plt.xlabel('P_gain')
    plt.ylabel('Error integral [V]')
    plt.show()

    plt.figure()
    plt.title('Peak error vs P_gain')
    plt.plot(stable_gbws, maximum_error)
    plt.xlabel('P_gain')
    plt.ylabel('Peak error [V]')
    plt.show()

    plt.figure()
    plt.title('RMSE vs P_gain')
    plt.plot(stable_gbws, rmse_sk)
    plt.xlabel('P_gain')
    plt.ylabel('RMSE error [V]')
    plt.show()
