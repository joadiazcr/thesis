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


# SSA low-pass filter model
def ssa_lpf(y0, t, u, ssa_bw):
    dydt = -ssa_bw * y0 + ssa_bw * u
    return dydt


# SSA saturation model
def ssa_sat(c, v_in):
    return v_in * (1.0 + (np.abs(v_in)**c))**(-1.0/c)


# SSA step response (low-pass filter + saturation)
def ssa(v_last, v_in, ssa_bw, c, t):
    v_out = odeint(ssa_lpf, v_last, [0, t], (v_in, ssa_bw))
    v_out_sat = ssa_sat(c, v_out[-1])  # Saturation
    return v_out_sat


# PI Controller
def PI(Kp, Ki, sp, x_in, sum_error, delta_t):
    error = sp - x_in

    sat = 1.278
    state = (sum_error + error * delta_t) * Ki
    scale = np.abs(state)/sat
    if scale > 1.0: state = state/scale 
    u = state + Kp * error
    scale = np.abs(u)/sat
    if scale > 1.0: u = u/scale 

    sum_error = sum_error + error * delta_t
    #u =  Kp * error + Ki * sum_error
    return u, error, sum_error

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
    Ql_pi = 1.0 / (1.0/Qg_pi + 1.0/Q0_pi + 1.0/Qprobe_pi)

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
        plt.vlines(1.0/bw, 0, np.abs(v_pi[-1]), 'r', '--')
        plt.hlines(2.0 * np.sqrt(RoverQ_pi * Qg_pi)*Kg, 0, t[-1], 'r', '--')
        plt.hlines(2.0 * np.sqrt(RoverQ_pi * Qg_pi)*Kg*(1.0-np.exp(-1)), 0, t[-1], 'r', '--')
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
        plt.vlines(1.0/bw, 0, np.abs(v_pi_I[-1]), 'r', '--')
        plt.hlines(Ql_pi*RoverQ_pi*Ib, 0, t[-1], 'r', '--')
        plt.hlines(Ql_pi*RoverQ_pi*Ib*(1.0-np.exp(-1)), 0, t[-1], 'r', '--')
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
        c = 50.0
        ssa_bw = 1e6
        power_max = 3.8e3

        Tmax = 1e-6
        Tstep = 1e-9
        trang = np.arange(0, Tmax, Tstep)
        nt = len(trang)

        v_out = np.zeros(nt, dtype=np.complex)
        v_out_sat = np.zeros(nt, dtype=np.complex)
        v_in = np.sqrt(power_max) * 0.6 / np.sqrt(power_max)  # Scale input signal (sqrt(W) -> Normalized units)

        for i in xrange(1, nt):
            v_last = v_out[i-1]
            v_out[i] = ssa(v_last, v_in, ssa_bw * 2.0 * np.pi, c, Tstep)  # SSA model (lpf + sat)
            v_out_sat[i] = v_out[i] * np.sqrt(power_max)
        plt.plot(trang, np.abs(v_out_sat), label='SSA Output', linewidth=3) # Scale output signal (sqrt(W) -> Normalized units)
        plt.ylim([0, 50])
        plt.ticklabel_format(style='sci', axis='x', scilimits=(1, 0))
        plt.title('SSA Test', fontsize=40, y=1.01)
        plt.xlabel('Time [s]', fontsize=30)
        plt.ylabel('Amplitude '+r'[$\sqrt{W}$]', fontsize=30)
        plt.legend(loc='upper right')

        plt.show()

        cs = [1, 2.5, 5.0, 15.8, 39.8]
        top_drive = 95  # %
        for c in cs:
            v_in = np.arange(0, 10, 1e-4)
            v_out = np.zeros(len(v_in), dtype=np.complex)
            for i in xrange(len(v_in)):
                v_out[i] = ssa_sat(c, v_in[i])
            plt.plot(v_in.real, v_out.real, label='c=%s' % c)

            # Sweep input
            for i in xrange(len(v_in)):
                v_out[i] = ssa_sat(c, v_in[i])
                if v_out[i].real >= (top_drive/100.0):
                    V_sat = v_in[i]
                    print V_sat
                    break

        plt.legend()
        plt.show()

    if args.f == 'phase':
        phase()

# I want to reproduce bode plots and stability analysis for nominal, high, critical and optimal(Using rezas method) configurations
