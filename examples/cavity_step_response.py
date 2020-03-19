import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

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

tf = 0.2 	# simulation time
nsteps = 500	# number of time steps
delta_t = tf/(nsteps-1)
ts = np.linspace(0, tf, nsteps) # linearly spaced time vector

# simulate step test operation
step = np.array([[0, 0]])

# PI mode cavity parameters
bw = 104  # where is this number coming from? Cavity bw 16Hz
RoverQ_pi = 1036.0
Qg_pi = 4.0 * 10**7
Q0_pi = 2.7 * 10**10
Qprobe_pi = 2.0 * 10**9

# set point
sp_amp = 350000.0
sp_phase = np.pi/8
#sp = [350000.0, 0]
sp = np.array([sp_amp, sp_phase])

# PI controller
# Nominal
stable_gbw = 20000
control_zero = 5000
#Kp = stable_gbw*2*np.pi/bw
#Ki = Kp*(2*np.pi*control_zero)
Kp = np.array([6/400000.0, 2])
Ki = np.array([1/800.0, 50])

sum_init = 0.0# Integral(sumation) initial value
e = np.array([[0, 0]])
ie = np.array([[0, 0]])

# simulate with ODEINT

foffset = 0
Kg = 1
Ib = 0

#v_s = np.zeros(nsteps)
v_s = np.array([[0]])
j = np.zeros(nsteps)
z0 = np.zeros(3)
 
for i in range(nsteps-1):

	v_s = np.append(v_s, (z0[0] + 1j*z0[1])*np.exp(1j*z0[2]))
        error = sp - [np.abs(v_s[i]), np.angle(v_s[i])]
        e = np.append(e, [error], axis=0)
        sum_init = sum_init + error * delta_t
	
	if Kp[0] == 0.0 and Ki[0] == 0.0:
		u = Kg
	else:
	        u = Kp * error + Ki * sum_init

	#ie[i+1] = sum_init
	ie = np.append(ie, [sum_init], axis = 0)
	if Kp[0] != 0.0 and Ki[0] != 0.0:
		step = np.append(step, [u], axis = 0)
	
	real = u[0] * np.cos(u[1])
	imag = u[0] * np.sin(u[1])

	args_pi = (RoverQ_pi, Qg_pi, Q0_pi, Qprobe_pi, bw, real, imag, Ib, foffset)
	z = odeint(cavity, z0, [0, delta_t], args = args_pi)
	z0 = z[-1] # take the last value of z

# plot results for Amplitude
fig = plt.figure()
fig.suptitle('Amplitude Control')
plt.subplot(2, 2 ,1)
plt.plot(ts, np.abs(v_s), 'b-', linewidth = 3, label = 'V')
plt.plot(ts, sp_amp * np.ones(nsteps), 'k--', linewidth = 2, label = 'Set point')
plt.ylabel(r'$| \vec V_{\rm acc}|$ [V]')
plt.legend()
plt.subplot(2, 2 ,2)
plt.plot(ts, step[:,0], 'r--', linewidth = 3, label = 'u')
plt.ylabel('Control  acction')
plt.legend()
plt.subplot(2, 2 ,3)
plt.plot(ts, e[:,0], 'b-', linewidth = 3, label = 'Error (SP_PV)')
plt.xlabel('Time [s]')
plt.legend()
plt.subplot(2, 2 ,4)
plt.plot(ts, ie[:,0], 'b-', linewidth = 3, label = 'Integral of error')
plt.xlabel('Time [s]')
plt.legend()

# plot results for phase
fig = plt.figure()
fig.suptitle('Phase Control')
plt.subplot(2, 2 ,1)
plt.plot(ts, np.angle(v_s), 'b-', linewidth = 3, label = 'V')
plt.plot(ts, sp_phase * np.ones(nsteps), 'k--', linewidth = 2, label = 'Set point')
plt.ylabel(r'$| \vec V_{\rm acc}|$ [V]')
plt.legend()
plt.subplot(2, 2 ,2)
plt.plot(ts, step[:,1], 'r--', linewidth = 3, label = 'u')
plt.ylabel('Control  acction')
plt.legend()
plt.subplot(2, 2 ,3)
plt.plot(ts, e[:,1], 'b-', linewidth = 3, label = 'Error (SP_PV)')
plt.xlabel('Time [s]')
plt.legend()
plt.subplot(2, 2 ,4)
plt.plot(ts, ie[:,1], 'b-', linewidth = 3, label = 'Integral of error')
plt.xlabel('Time [s]')
plt.legend()
plt.show()
