import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import cavity_model


def meas_noise(psd, bw, grad, k):
    rms = 1.5 * np.sqrt(0.5 * 10**(psd/10.0) * bw) * grad * k
    return rms


def plot(t, cav_v, sp, fwd, show_fwd=False, show_lim=False,
         x_lim=None, y_lim=None, plot_type='amp'):
    if plot_type == 'amp':
        plt.plot(t, np.abs(cav_v), 'b-', linewidth=2, label='Cavity Voltage (Probe)')
        plt.plot(t, sp * np.ones(len(t)), 'k--', linewidth=2, label='Set point')
        if show_fwd:
            plt.plot(t, np.abs(fwd[:, 0]), 'r-', linewidth=2, label='U (Forward)')
        plt.ylabel(r'$| \vec V_{\rm acc}|$ [V]')

    if plot_type == 'phase':
        plt.plot(t, np.angle(cav_v), 'b-', linewidth=2, label='Cavity Voltage (Probe)')
        plt.plot(t, sp * np.zeros(len(t)), 'k--', linewidth=2, label='Set point')
        if show_fwd:
            plt.plot(t, np.angle(fwd[:, 0]), 'r-', linewidth=2, label='U (Forward)')
        plt.ylabel('Phase [degress]')

    if plot_type == 'cartesian':
        plt.plot(t, np.real(cav_v), 'b-', linewidth=2, label='Cavity Voltage Real(Probe)')
        plt.plot(t, np.imag(cav_v), 'b--', linewidth=2, label='Cavity Voltage Imag(Probe)')
        if show_fwd:
            plt.plot(t, np.real(fwd[:, 0]), 'r-', linewidth=2, label='U Real (Forward)')
            plt.plot(t, np.imag(fwd[:, 0]), 'r--', linewidth=2, label='U Imag (Forward)')
        plt.ylabel(r'$| \vec V_{\rm acc}|$ [V]')
    
    if show_lim:
        if plot_type == 'amp':
            plt.plot(t, nom_grad*np.ones(len(t))*(1+1e-4), label='Upper limit', linewidth=2, color='m')
            plt.plot(t, nom_grad*np.ones(len(t))*(1-1e-4), label='Lower limit', linewidth=2, color='y')
            plt.axhspan(nom_grad/1.00005, nom_grad*1.00005, color='blue', alpha=0.2)
        if plot_type == 'phase':
            plt.plot(t, 1e-2*np.ones(len(t)), label='Upper limit', linewidth=2, color='m')
            plt.plot(t, -1e-2*np.ones(len(t)), label='Lower limit', linewidth=2, color='y')
            plt.axhspan(-4e-3, 4e-3, color='blue', alpha=0.2)

    if x_lim:
        plt.xlim(x_lim)

    if y_lim:
        plt.ylim(y_lim)

    plt.xlabel('Time [ms]')
    plt.legend()
    plt.show()


def cavity_step(tf, nom_grad, feedforward, llrf_noise, psd, beam, beam_start, beam_end, detuning, detuning_start, detuning_end):
	# Pi mode cavity parameters
	bw = 104  # where is this number coming from? Cavity bw 16Hz
	RoverQ_pi = 1036.0
	Qg_pi = 4.0 * 10**7
	Q0_pi = 2.7 * 10**10
	Qprobe_pi = 2.0 * 10**9

	# Cavity conditions
	Kg = 1
	Ib = 0
	f_os = 0.0
	noise_bw = 53000.0 * 2.0 * np.pi  # [Hz] Noise shaping filter. Not yet implemented

	# SSA parameters
	c = 5.0
	ssa_bw = 1e6
	power_max = 3.8e3

	# PI controller - Nominal Configuration
	stable_gbw = 20000
	control_zero = 5000
	Kp_a = stable_gbw*2*np.pi/bw
	Ki_a = Kp_a*(2*np.pi*control_zero)
	Kp_p = 1.0
	Ki_p = 1.0
	Kp = np.array([Kp_a, Kp_p])
	Ki = np.array([Ki_a, Ki_p])

	# Simulation time variables
	t_step = 1e-6
	ts = np.arange(0, tf, t_step)

	# set point
	fund_k_probe = 6.94e-7
	fund_k_drive = 407136.3407999831
	fund_k_beam = 40568527918.78173+0j
	sp_amp = nom_grad * fund_k_probe # 11.32
	sp_phase = 0.0
	sp = np.array([sp_amp, sp_phase])

	# Initializing simulation variables
	sum_init = np.zeros(2,  dtype='complex')  # Integral(sumation) initial value
	error = np.zeros(2, dtype='complex') # Control error initial value
	e = np.array([[0, 0]])  # Array to store and plot control error
	ie = np.array([[0, 0]])  # Array to store and plot integral of the eror
	v_s = np.array([[0]], dtype='complex')  # Mode cavity voltage array
	z0 = np.zeros(3) # Initial conditions of cavity voltage (real, imaginary, phase)
	u = np.zeros(2, dtype='complex') # Control signal initial value
	v_last = 0.0 # Last state of SSA output
	step = np.array([[0, 0]]) # Array to store and plot control signal
	
	# beam current
	beam_current = 100.0 * 10.0 ** (-6)
	# feedforward
	feed_forward = 0.99*np.sqrt(2)*beam_current*fund_k_beam/fund_k_drive + 0j
	# LLRF noise bw
	bw_llrf_ns = 0.5 * 1.0/(t_step) # / 10.0  # the "/10" should not be there, check that

	for i in range(len(ts)-1):

		vs_amp =  z0[0] + 1j*z0[1]
		if llrf_noise == True:
			rms = meas_noise(psd, bw_llrf_ns, nom_grad, fund_k_probe)
			vs_amp = cavity_model.gauss_noise(vs_amp, 0.0, rms/fund_k_probe) # apply gaussian noise to cavity voltage
		v_s = np.append(v_s, (vs_amp)*np.exp(1j*z0[2]))

		# Apply loop delay
		# This loop delay is 1us, since my time step is 1us, i dont think this should be included

		# Apply noise shaping low-pass filter
		j = cavity_model.odeintz(cavity_model.noise_lpf_complex, v_s[i-1], [0, t_step], args=(v_s[i], 53000.0 * 2.0 * np.pi))  # Low-pass filter
		v_s[i] = j[-1]


		if Kp[0] == 0.0 and Ki[0] == 0.0:
			u = Kg
		else:
			u[0], error[0], sum_init[0] = cavity_model.PI(Kp[0], Ki[0], sp[0], v_s[i] * fund_k_probe, sum_init[0], t_step)
			u[1] = 0.0
			e = np.append(e, [error], axis=0)
			ie = np.append(ie, [sum_init], axis=0)

		u[0], v_last = cavity_model.ssa(v_last, u[0], ssa_bw * 2.0 * np.pi, c, t_step, power_max)

		if beam == True:
			if (i * t_step) > beam_start and (i * t_step) < beam_end: 
				Ib = 100.0 * 10.0 ** (-12) / (10**(-6)) # 100 mA
				if feedforward == True:
					u[0] = u[0] + feed_forward
			else:
				Ib = 0.0

		if detuning == True:
			if (i * t_step) > detuning_start and (i * t_step) < detuning_end: 
				f_os = 10.0 * np.sin(2.0 * np.pi * 100 * i * t_step)
			else:
				f_os = 0.0 

		if Kp[0] != 0.0 and Ki[0] != 0.0:
			step = np.append(step, [u], axis=0)

		real = np.real(u[0])
		imag = np.imag(u[0])

		args_pi = (RoverQ_pi, Qg_pi, Q0_pi, Qprobe_pi, bw, real, imag, Ib, f_os)
		z = odeint(cavity_model.model_complex, z0, [0, t_step], args=args_pi)
		z0 = z[-1] # take the last value of z

	return ts, v_s, nom_grad, step * fund_k_drive


if __name__ == "__main__":
    des = 'Cavity step response simulation including models of the SSA and\
	       the PI controller, and perturbations related with measurement\
		   noise, beam loading and cavity detuning'
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('-b', "--beam", action="store_true",
                        help='Include beam')
    parser.add_argument('-d', "--detuning", action="store_true",
                        help='Include detuning')
    parser.add_argument('-n', "--noise", action="store_true",
                        help='Include measurement noise')
    parser.add_argument('-f', "--feed_forward", action="store_true",
                        help='Feed forward ON')
    
    args = parser.parse_args()

    beam = args.beam
    detuning = args.detuning
    noise = args.noise
    feedforward = args.feed_forward

    # Some settings for the simulation
    tf = 0.02 # Total simulation time
    nom_grad = 1.6301e7 # V/m. Nominal gradient of LCLS-II cavity
    noise_psd = -149
    beam_start = 0.015
    beam_end = 0.017
    detuning_start = 0.0
    detuning_end = 0.03

    if noise and not beam and not detuning:
        tf = 0.03 # Total simulation time
        t, cav_v, sp, fwd = cavity_step(tf, nom_grad, feedforward, noise, noise_psd, beam, beam_start, beam_end, detuning, detuning_start, detuning_end)    
        plot(t, cav_v, sp, fwd, show_fwd=True)
        plot(t, cav_v, sp, fwd, show_lim=True, x_lim=[0.012, 0.02], y_lim=[nom_grad-10e3, nom_grad+10e3])
        plot(t, cav_v, sp, fwd, plot_type='phase', show_lim=True, x_lim=[0.012, 0.02], y_lim=[-6.25e-2, 6.25e-2])
    
    if beam and not noise and not detuning:
        t, cav_v, sp, fwd = cavity_step(tf, nom_grad, feedforward, noise, noise_psd, beam, beam_start, beam_end, detuning, detuning_start, detuning_end)    
        plot(t, cav_v, sp, fwd, show_fwd=True)
        plot(t, cav_v, sp, fwd, show_fwd=True, x_lim=[0.0145, 0.0175], y_lim=[14e6, 23e6])
        plot(t, cav_v, sp, fwd, show_fwd=True, x_lim=[0.01495, 0.0153], y_lim=[sp-10e3, sp+10e3], show_lim=True)
        plot(t, cav_v, sp, fwd, show_fwd=True, x_lim=[0.01495, 0.017], y_lim=[sp-10e3, sp+10e3], show_lim=True)

    if detuning:
        tf = 0.03 # Total simulation time
        beam_start = 0.015
        beam_end = 0.025
        t, cav_v, sp, fwd = cavity_step(tf, nom_grad, feedforward, noise, noise_psd, beam, beam_start, beam_end, detuning, detuning_start, detuning_end)    
        plot(t, cav_v, sp, fwd, show_fwd=True)
        plot(t, cav_v, sp, fwd, show_fwd=True, plot_type='cartesian', x_lim=[0.01495, 0.025])
        plot(t, cav_v, sp, fwd, show_fwd=True, x_lim=[0.01495, 0.025], y_lim=[sp-10e3, sp+10e3], show_lim=True)
        plot(t, cav_v, sp, fwd, plot_type='phase', show_lim=True, x_lim=[0.01495, 0.025], y_lim=[-6.25e-2, 6.25e-2])

    if not beam and not noise and not detuning:
        tf = 0.03 # Total simulation time
        t, cav_v, sp, fwd = cavity_step(tf, nom_grad, feedforward, noise, noise_psd, beam, beam_start, beam_end, detuning, detuning_start, detuning_end)    
        plot(t, cav_v, sp, fwd, show_fwd=True)
