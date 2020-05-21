import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import cavity_model


# Pi mode cavity parameters
bw = 104  # where is this number coming from? Cavity bw 16Hz
RoverQ_pi = 1036.0
Qg_pi = 4.0 * 10**7
Q0_pi = 2.7 * 10**10
Qprobe_pi = 2.0 * 10**9

# Cavity conditions
foffset = 0.0
foffset_s = 0.2 
Kg = 1
Ib = 0

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
tf = 0.05  # Total simulation time
t_step = 1e-6
ts = np.arange(0, tf, t_step)

# set point
fund_k_probe = 6.94e-7
fund_k_drive = 407136.3407999831
nom_grad = 1.6301e7 # V/m. Nominal gradient of LCLS-II cavity
sp_amp = nom_grad * fund_k_probe # 11.32
sp_phase = 0.0
sp = np.array([sp_amp, sp_phase])

# Initializing simulation variables
sum_init = np.zeros(2)  # Integral(sumation) initial value
error = np.zeros(2) # Control error initial value
e = np.array([[0, 0]])  # Array to store and plot control error
ie = np.array([[0, 0]])  # Array to store and plot integral of the eror
v_s = np.array([[0]])  # Mode cavity voltage array
z0 = np.zeros(3) # Initial conditions of cavity voltage (real, imaginary, phase)
u = np.zeros(2) # Control signal initial value
v_last = 0.0 # Last state of SSA output
step = np.array([[0, 0]]) # Array to store and plot control signal

for i in range(len(ts)-1):

	vs_amp =  z0[0] + 1j*z0[1]
	vs_amp = cavity_model.gauss_noise(vs_amp, 0.0, 0.000151/fund_k_probe) # apply gaussian noise to cavity voltage
	v_s = np.append(v_s, (vs_amp)*np.exp(1j*z0[2]))

	if (i * t_step) < foffset_s: 
		f_os = 0.0
	else:
		f_os = foffset 
	
	if Kp[0] == 0.0 and Ki[0] == 0.0:
		u = Kg
	else:
		u[0], error[0], sum_init[0] = cavity_model.PI(Kp[0], Ki[0], sp[0], np.abs(v_s[i]) * fund_k_probe, sum_init[0], t_step)
		u[1] = 0.0
		e = np.append(e, [error], axis=0)
		ie = np.append(ie, [sum_init], axis=0)

	u[0], v_last = cavity_model.ssa(v_last, u[0], ssa_bw * 2.0 * np.pi, c, t_step, power_max)

	if Kp[0] != 0.0 and Ki[0] != 0.0:
		step = np.append(step, [u], axis=0)

	real = u[0] * np.cos(u[1])
	imag = u[0] * np.sin(u[1])

	args_pi = (RoverQ_pi, Qg_pi, Q0_pi, Qprobe_pi, bw, real, imag, Ib, f_os)
	z = odeint(cavity_model.model_complex, z0, [0, t_step], args=args_pi)
	z0 = z[-1] # take the last value of z


# Plot results
plt.plot(ts, np.abs(v_s), 'b-', linewidth=2, label='Cavity Voltage (Probe)')
plt.plot(ts, nom_grad * np.ones(len(ts)), 'k--', linewidth=2, label='Set point')
plt.plot(ts, step[:, 0] * fund_k_drive, 'r-', linewidth=2, label='U')
plt.ylabel(r'$| \vec V_{\rm acc}|$ [V]')
plt.legend()
plt.show()

# plot results for Amplitude
fig = plt.figure()
fig.suptitle('Amplitude Control')
plt.subplot(2, 2, 1)
plt.plot(ts, np.abs(v_s), 'b-', linewidth=1, label='V')
plt.plot(ts, nom_grad * np.ones(len(ts)), 'k--', linewidth=2, label='Set point')
plt.ylabel(r'$| \vec V_{\rm acc}|$ [V]')
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(ts, step[:, 0], 'r-', linewidth=1, label='u')
plt.ylabel('Control  acction')
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(ts, e[:, 0], 'b-', linewidth=3, label='Error (SP_PV)')
plt.xlabel('Time [s]')
plt.legend()
#plt.show()

# plot results for phase
fig = plt.figure()
fig.suptitle('Phase Control')
plt.subplot(2, 2, 1)
plt.plot(ts, np.angle(v_s), 'b-', linewidth=3, label='V')
plt.plot(ts, sp_phase * np.ones(len(ts)), 'k--', linewidth=2, label='Set point')
plt.ylabel(r'$| \vec V_{\rm acc}|$ [V]')
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(ts, step[:, 1], 'r--', linewidth=3, label='u')
plt.ylabel('Control  acction')
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(ts, e[:, 1], 'b-', linewidth=3, label='Error (SP_PV)')
plt.xlabel('Time [s]')
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(ts, ie[:, 1], 'b-', linewidth=3, label='Integral of error')
plt.xlabel('Time [s]')
plt.legend()
#plt.show()
