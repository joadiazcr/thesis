import numpy as np
import argparse
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import PIcontrol as pi
import json

# Model for the accelerating voltage inside a cavity
# based on CMOC Equation 3.13
# Function that returns dS/dt

# Cavity model with Kg and Ib real numbers
def model(z, t, RoverQ, Qg, Q0, Qprobe, bw, Kg, Ib,
          foffset, Kg_s=0.0, foffset_s=0.0):
    if t < Kg_s: Kg = 0.0
    if t < foffset_s: foffset = 0.0

    Rg = RoverQ * Qg
    Kdrive = 2 * np.sqrt(Rg)
    Ql = 1.0 / (1.0/Qg + 1.0/Q0 + 1.0/Qprobe)
    K_beam = RoverQ * Ql
    w_d = 2 * np.pi * foffset

    a = Kdrive * bw
    b = K_beam * bw
    c = bw

    yr, yi, theta = z

    dthetadt = w_d
    dydt_r = (a * Kg - b * Ib)*np.cos(theta) - (c * yr)
    dydt_i = -(a * Kg - b * Ib)*np.sin(theta) - (c * yi)
    return [dydt_r, dydt_i, dthetadt]


# Cavity model with Kg complex number
def model_b(z, t, RoverQ, Qg, Q0, Qprobe, bw, Kg_r, Kg_i, Ib,
          foffset, Kg_s=0.0, foffset_s=0.0):
    if t < Kg_s: 
	Kg_r = 0.0
	Kg_i = 0.0
    if t < foffset_s: foffset = 0.0

    Rg = RoverQ * Qg
    Kdrive = 2 * np.sqrt(Rg)
    Ql = 1.0 / (1.0/Qg + 1.0/Q0 + 1.0/Qprobe)
    K_beam = RoverQ * Ql
    w_d = 2 * np.pi * foffset

    a = Kdrive * bw
    b = K_beam * bw
    c = bw

    yr, yi, theta = z

    dthetadt = w_d
    dydt_r = (a * Kg_r - b * Ib)*np.cos(theta) - (c * yr) + a * Kg_i * np.sin(theta)
    dydt_i = -(a * Kg_r - b * Ib)*np.sin(theta) - (c * yi) + a * Kg_i * np.cos(theta)
    return [dydt_r, dydt_i, dthetadt]


def step_response(args, t):
    # initial condition
    z0 = [0, 0, 0]  # y_r, y_i, theta

    y = odeint(model_b, z0, t, args)
    v = (y[:, 0] + 1j*y[:, 1])*np.exp(1j*y[:, 2])

    return v


def ssa():
    cs = [1, 2.5, 6.3, 15.8, 39.8]
    for c in cs:
        v_in = np.linspace(0, 10, 1000)  # 0.2 seconds
        v_out = v_in * (1 + v_in**c)**(-1/c)
        plt.plot(v_in, v_out, label='c=%s' % c)
    plt.legend()
    plt.show()


def phase():
    t = np.linspace(0, 1, 1000)
    f = 3.0
    theta = 0.0
    v_in = np.exp(1j * (2.0 * np.pi * f * t + theta))
    shift = [0, np.pi/4.0, np.pi/2.0, 3 * np.pi/4.0, np.pi]
    for sh in shift:
        v_out = v_in * np.exp(1j * sh)
        plt.plot(t, np.real(v_out), label='shift=%s' % sh)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cavity Simulation')
    parser.add_argument('-f', type=str, choices=['stp', 'pi', 'ssa', 'phase'],
                        help='Function: spt for Step Response,\
                        pi for PI controller,\
                        ssa for SSA saturation curve\
                        or phase phase shift')

    args = parser.parse_args()

    # initial condition for the ODE
    z0 = [0, 0, 0]  # y_r, y_i, theta

    # time points
    t = np.linspace(0, 0.2, 1000)  # 0.2 seconds

    bw = 104  # where is this number coming from? Cavity bw 16Hz

    # values for the different modes come from https://github.com/BerkeleyLab/Global-Feedback-Simulator/blob/release/source/configfiles/LCLS-II/LCLS-II_accelerator.json
    # Pi mode
    RoverQ_pi = 1036.0
    Qg_pi = 4.0 * 10**7
    Q0_pi = 2.7 * 10**10
    Qprobe_pi = 2.0 * 10**9

    # 8pi/9 mode
    RoverQ_8pi9 = 20
    Qg_8pi9 = 8.1 * 10**7
    Q0_8pi9 = 1.0 * 10**10
    Qprobe_8pi9 = 2.0 * 10**9

    # pi/9 mode
    RoverQ_pi9 = 10
    Qg_pi9 = 8.1 * 10**7
    Q0_pi9 = 1.0 * 10**10
    Qprobe_pi9 = 2.0 * 10**9

    if args.f == 'stp':
        # solve ODEs for cavity response to a step funtion on the RF drive signal
        foffset = 0
        Kg = 1
        Ib = 0
        args_pi = (RoverQ_pi, Qg_pi, Q0_pi, Qprobe_pi, bw, Kg, 0, Ib, foffset)
        args_8pi9 = (RoverQ_8pi9, Qg_8pi9, Q0_8pi9, Qprobe_8pi9, bw, Kg, 0, Ib, foffset)
        args_pi9 = (RoverQ_pi9, Qg_pi9, Q0_pi9, Qprobe_pi9, bw, Kg, 0, Ib, foffset)
        v_pi = step_response(args_pi, t)
        v_8pi9 = step_response(args_8pi9, t)
        v_pi9 = step_response(args_pi9, t)

        # plot results
        plt.plot(t, np.abs(v_pi), label='pi')
        print "Pi Drive coupling = %s" % max(np.abs(v_pi))
        plt.plot(t, np.abs(v_8pi9), label='8pi/9')
        print "8pi/9 Drive coupling = %s" % max(np.abs(v_8pi9))
        plt.plot(t, np.abs(v_pi9), label='pi/9')
        print "pi/9 Drive coupling = %s" % max(np.abs(v_pi9))
        plt.title("Cavity Step Response: Drive @ "+r'$\omega_{\mu}$')
        plt.xlabel('Time [s]')
        plt.ylabel(r'$| \vec V_{\rm acc}|$ [V]')
        plt.legend()
        plt.show()

        # Solve ODEs for cavity response to a step funtion on the beam input signal
        Kg = 0
        # current equivalent to 1pC of charge. Step size is 1us for CMOC simulations
        Ib = 1 * 10 ** (-12) / (10**(-6))
        args_pi = (RoverQ_pi, Qg_pi, Q0_pi, Qprobe_pi, bw, Kg, 0, Ib, foffset)
        args_8pi9 = (RoverQ_8pi9, Qg_8pi9, Q0_8pi9, Qprobe_8pi9, bw, Kg, 0, Ib, foffset)
        args_pi9 = (RoverQ_pi9, Qg_pi9, Q0_pi9, Qprobe_pi9, bw, Kg, 0, Ib, foffset)
        v_pi_I = step_response(args_pi, t)
        v_8pi9_I = step_response(args_8pi9, t)
        v_pi9_I = step_response(args_pi9, t)

        # plot results
        plt.plot(t, np.abs(v_pi_I), label='pi')
        print "Pi beam coupling = %s" % max(np.abs(v_pi_I))
        plt.plot(t, np.abs(v_8pi9_I), label='8pi/9')
        print "8Pi/9 beam coupling = %s" % max(np.abs(v_8pi9_I))
        plt.plot(t, np.abs(v_pi9_I), label='pi/9')
        print "Pi beam coupling = %s" % max(np.abs(v_pi9_I))
        plt.title("Cavity Step Response: 1 pC Beam step")
        plt.xlabel('Time [s]')
        plt.ylabel(r'$| \vec V_{\rm acc}|$ [V]')
        plt.legend()
        plt.show()

        # Testing frequency offsets
        # time points
        Kg = 1
        Ib = 0
        foffset = [0, 10, 20, 100]
        for f in foffset:
            args_pi = (RoverQ_pi, Qg_pi, Q0_pi, Qprobe_pi, bw, Kg, 0, Ib, f)
            v_pi_f = step_response(args_pi, t)

            # plot results
            plt.plot(np.real(v_pi_f), np.imag(v_pi_f), label='basis offset=%s Hz' % f)

        plt.title("Cavity Step Response: Drive @ "+r'$\omega_{ref}$')
        plt.xlabel(r'$\Re (\vec V_{\mu})$ [V]')
        plt.ylabel(r'$\Im (\vec V_{\mu})$ [V]')
        plt.ylim([-0.5e5, 3e5])
        plt.xlim([-0.5e5, 5e5])
        plt.axes().set_aspect('equal')
        plt.legend()
        plt.show()

        # Testing fill and step on detune frequency
        nt = 10000
        t = np.linspace(0, 0.5, nt)  # 0.5 seconds
        Kg = 1.0
        Ib = 0.0
        Kg_s = 0.05
        foffset = [100.0, 200.0]
        foffset_s = 0.2

        fig, ax1 = plt.subplots()
        fig2, ax3 = plt.subplots()
        ax2 = ax1.twinx()
        lns = []
        colors = ['c', 'm']

        for idx, f in enumerate(foffset):
            args_pi = (RoverQ_pi, Qg_pi, Q0_pi, Qprobe_pi, bw, Kg, 0, Ib, f, Kg_s, foffset_s)
            v_pi_d = step_response(args_pi, t)

            # plot results for magnitude
            lns += ax1.plot(t, np.abs(v_pi_d), label='Cavity Field ('+r'$\Delta f_1$ = %s Hz' % f+')')
            ax3.plot(t, np.unwrap(np.angle(v_pi_d)), label='Cavity Field ('+r'$\Delta f_1$ = %s Hz' % f + ')')

            delta_f = np.ones(nt)*f
            delta_f[0:int(nt*0.4)] = 0.0
            lns += ax2.plot(t, delta_f, '-%s' % colors[idx], label=r'$\Delta f_1$ = %s Hz' % f)

        drive_in = np.ones(nt)
        drive_in[0:int(nt*0.1)] = 0.0

        lns += ax1.plot(t, drive_in*max(np.abs(v_pi_d)), label='Drive')
        labs = [l.get_label() for l in lns]

        ax3.set_title("Cavity Step Response: 1 pC Beam step")
        plt.title("Cavity Step Response: 1 pC Beam step")
        ax1.set_xlabel('Time [s]')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel(r'$\angle \vec V_{\mu}$ [rad]')
        ax2.set_ylabel('Frequency [Hz]')
        ax1.set_ylabel(r'$| \vec V_{\rm acc}|$ [V]')
        ax1.set_ylim(0, 6e5)
        ax2.set_ylim(0, foffset[1]*1.5)
        ax1.legend(lns, labs, loc=0)
        plt.show()

    if args.f == 'pi':
        f = np.linspace(1, 1e6, 1e6)  # Frequency from 1 to 10^6 Hz
        s = 1j*2.0*np.pi*f  # Laplace operator s=jw=j*2*pi*f
        wc = 104.3  # rad/s. bandwidth? where is this number coming from? from 16.6 Hz cavity bandwidth
        cavity = 1.0/(1.0+s/wc)  # Cavity transfer function. Cavity acts as a low pass filter
        configurations = ['nominal', 'high', 'hobicat']
        for conf in configurations:
            with open("%s.json" % conf, "r") as read_file:
                data = json.load(read_file)
            stable_gbw = data['stable_gbw']
            ssa_bw = data['ssa_bw']
            control_zero = data['control_zero']
            pi.pid(f, s, wc, cavity, ssa_bw, stable_gbw, control_zero, conf, 1)

    if args.f == 'ssa':
        ssa()

    if args.f == 'phase':
        phase()

# I want to reproduce bode plots and stability analysis for nominal, high, critical and optimal(Using rezas method) configurations
