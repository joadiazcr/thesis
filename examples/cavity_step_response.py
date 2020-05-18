import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import cavity_model

# Define Cavity model
def cavity(z, t, RoverQ, Qg, Q0, Qprobe, bw, Kg_r, Kg_i, Ib,
           foffset):

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
	return dydt_r, dydt_i, dthetadt


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
ssa_bw = 53000.0 * 2.0 * np.pi  # [Hz]

# SSA parameters
c = 5.0
ssa_bw = 1e6
power_max = 3.8e3

# PI controller
# Nominal Configuration
stable_gbw = 20000
control_zero = 5000
Kp_a = stable_gbw*2*np.pi/bw
Ki_a = Kp_a*(2*np.pi*control_zero)
Kp_p = 1.0
Ki_p = 10.0
Kp = np.array([3500.0, Kp_p])
Ki = np.array([1/800.0, Ki_p])

# Simulation time variables
tf = 0.5  # Total simulation time
nsteps = 20000  # number of time steps
delta_t = tf/(nsteps-1)
ts = np.linspace(0, tf, nsteps)  # linearly spaced time vector

# simulate step test operation
step = np.array([[0, 0]])

# set point
sp_amp = 3000000
sp_phase = 0.0
sp = np.array([sp_amp, sp_phase])

# Initializing simulation variables
sum_init = 0.0  # Integral(sumation) initial value
e = np.array([[0, 0]])  # error initial value
ie = np.array([[0, 0]])  # Integral of the eror initial value
v_s = np.array([[0]])  # Mode cavity voltage
z0 = np.zeros(3)
u = np.zeros(2)
error = np.zeros(2)
sum_init = np.zeros(2)

for i in range(nsteps-1):

	v_s = np.append(v_s, (z0[0] + 1j*z0[1])*np.exp(1j*z0[2]))

	if (i * delta_t) < foffset_s: 
		f_os = 0.0
	else:
		f_os = foffset 
	
	if Kp[0] == 0.0 and Ki[0] == 0.0:
		u = Kg
	else:
	        u[0], error[0], sum_init[0] = cavity_model.PI(Kp[0], Ki[0], sp[0], np.abs(v_s[i]), sum_init[0], delta_t)
                u[1] = 0.0
                e = np.append(e, [error], axis=0)
                ie = np.append(ie, [sum_init], axis=0)
        v_last = step[-1, 0]

	u[0], v_last = cavity_model.ssa(v_last, u[0], ssa_bw * 2.0 * np.pi, c, delta_t, power_max)

	if Kp[0] != 0.0 and Ki[0] != 0.0:
		step = np.append(step, [u], axis=0)

	real = u[0] * np.cos(u[1])
	imag = u[0] * np.sin(u[1])

	args_pi = (RoverQ_pi, Qg_pi, Q0_pi, Qprobe_pi, bw, real, imag, Ib, f_os)
	z = odeint(cavity_model.model_complex, z0, [0, delta_t], args=args_pi)
	z0 = z[-1] # take the last value of z

# plot results for Amplitude
fig = plt.figure()
fig.suptitle('Amplitude Control')
plt.subplot(2, 2, 1)
plt.plot(ts, np.abs(v_s), 'b-', linewidth=3, label='V')
plt.plot(ts, sp_amp * np.ones(nsteps), 'k--', linewidth=2, label='Set point')
plt.ylabel(r'$| \vec V_{\rm acc}|$ [V]')
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(ts, step[:, 0], 'r--', linewidth=3, label='u')
plt.ylabel('Control  acction')
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(ts, e[:, 0], 'b-', linewidth=3, label='Error (SP_PV)')
plt.xlabel('Time [s]')
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(ts, ie[:, 0], 'b-', linewidth=3, label='Integral of error')
plt.xlabel('Time [s]')
plt.legend()
plt.show()

# plot results for phase
fig = plt.figure()
fig.suptitle('Phase Control')
plt.subplot(2, 2, 1)
plt.plot(ts, np.angle(v_s), 'b-', linewidth=3, label='V')
plt.plot(ts, sp_phase * np.ones(nsteps), 'k--', linewidth=2, label='Set point')
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
