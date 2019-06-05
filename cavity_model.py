import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Model for the accelerating voltage inside a cavity
# based on CMOC Equation 3.13
# Function that returns dS/dt
def model(z, t, RoverQ, Qg, Q0, Qprobe, bw, Kg, Ib, foffset):
    Rg = RoverQ * Qg
    Kdrive = 2 * np.sqrt(Rg)
    bw = 104
    Ql = 1.0 / (1.0/Qg + 1.0/Q0 + 1.0/Qprobe)
    K_beam = RoverQ * Ql
    w_d = 2 * np.pi * foffset

    a = Kdrive * bw
    b = K_beam * bw
    c = bw

    yr, yi, theta = z

    dthetadt = w_d
    dydt_r = (a * Kg + b * Ib)*np.cos(theta) - (c * yr)
    dydt_i = -(a * Kg + b * Ib)*np.sin(theta) - (c * yi)
    return [dydt_r, dydt_i, dthetadt]


# initial condition
z0 = [0, 0, 0]  # y_r, y_i and theta

# time points
t = np.linspace(0, 0.2, 1000)  # 0.2 seconds

# solve ODEs for cavity response to a step funtion on the RF drive signal
bw = 104
foffset = 0
Kg = 1
Ib = 0
# Pi mode
RoverQ_pi = 1036
Qg_pi = 4.0 * 10**7
Q0_pi = 2.7 * 10**10
Qprobe_pi = 2.0 * 10**9
args_pi = (RoverQ_pi, Qg_pi, Q0_pi, Qprobe_pi, bw, Kg, Ib, foffset)
y_pi = odeint(model, z0, t, args_pi)
v_pi = (y_pi[:, 0] + 1j*y_pi[:, 1])*np.exp(1j*y_pi[:, 2])

# 8pi/9 mode
RoverQ_8pi9 = 20
Qg_8pi9 = 8.1 * 10**7
Q0_8pi9 = 1.0 * 10**10
Qprobe_8pi9 = 2.0 * 10**9
args_8pi9 = (RoverQ_8pi9, Qg_8pi9, Q0_8pi9, Qprobe_8pi9, bw, Kg, Ib, foffset)
y_8pi9 = odeint(model, z0, t, args_8pi9)
v_8pi9 = (y_8pi9[:, 0] + 1j*y_8pi9[:, 1])*np.exp(1j*y_8pi9[:, 2])

# pi/9 mode
RoverQ_pi9 = 10
Qg_pi9 = 8.1 * 10**7
Q0_pi9 = 1.0 * 10**10
Qprobe_pi9 = 2.0 * 10**9
args_pi9 = (RoverQ_pi9, Qg_pi9, Q0_pi9, Qprobe_pi9, bw, Kg, Ib, foffset)
y_pi9 = odeint(model, z0, t, args_pi9)
v_pi9 = (y_pi9[:, 0] + 1j*y_pi9[:, 1])*np.exp(1j*y_pi9[:, 2])

# plot results
plt.plot(t, np.abs(v_pi), label='pi')
plt.plot(t, np.abs(v_8pi9), label='8pi/9')
plt.plot(t, np.abs(v_pi9), label='pi/9')
plt.title("Cavity Step Response: Drive @ "+r'$\omega_{\mu}$')
plt.xlabel('Time [s]')
plt.ylabel(r'$| \vec V_{\rm acc}|$ [V]')
plt.legend()
plt.show()

# Solve ODEs for cavity response to a step funtion on the RF drive signal
Kg = 0
# current equivalent to 1pC of charge. Step size is 1us for CMOC simulations
Ib = 1 * 10 ** (-12) / (10**(-6))
# Pi mode
args_pi = (RoverQ_pi, Qg_pi, Q0_pi, Qprobe_pi, bw, Kg, Ib, foffset)
y_pi_I = odeint(model, z0, t, args_pi)
v_pi_I = (y_pi_I[:, 0] + 1j*y_pi_I[:, 1])*np.exp(1j*y_pi_I[:, 2])

# 8pi/9 mode
args_8pi9 = (RoverQ_8pi9, Qg_8pi9, Q0_8pi9, Qprobe_8pi9, bw, Kg, Ib, foffset)
y_8pi9_I = odeint(model, z0, t, args_8pi9)
v_8pi9_I = (y_8pi9_I[:, 0] + 1j*y_8pi9_I[:, 1])*np.exp(1j*y_8pi9_I[:, 2])

# pi/9 mode
args_pi9 = (RoverQ_pi9, Qg_pi9, Q0_pi9, Qprobe_pi9, bw, Kg, Ib, foffset)
y_pi9_I = odeint(model, z0, t, args_pi9)
v_pi9_I = (y_pi9_I[:, 0] + 1j*y_pi9_I[:, 1])*np.exp(1j*y_pi9_I[:, 2])

# plot results
plt.plot(t, np.abs(v_pi_I), label='pi')
plt.plot(t, np.abs(v_8pi9_I), label='8pi/9')
plt.plot(t, np.abs(v_pi9_I), label='pi/9')
plt.title("Cavity Step Response: 1 pC Beam step")
plt.xlabel('Time [s]')
plt.ylabel(r'$| \vec V_{\rm acc}|$ [V]')
plt.legend()
plt.show()


# Testing frequency offsets
# time points
t = np.linspace(0, 1, 10000)  # 0.2 seconds
foffset = [0, 10, 20, 100]
for f in foffset:
	args_pi = (RoverQ_pi, Qg_pi, Q0_pi, Qprobe_pi, bw, Kg, Ib, f)
	y_pi_f = odeint(model, z0, t, args_pi)
	v_pi_f = (y_pi_f[:, 0] + 1j*y_pi_f[:, 1])*np.exp(1j*y_pi_f[:, 2])

	# plot results
	plt.plot(np.real(v_pi_f), np.imag(v_pi_f), label='basis offset=%s' %f)

plt.title("Cavity Step Response: Drive @ "+r'$\omega_{ref}$')
plt.xlabel(r'$\Re (\vec V_{\mu})$ [V]')
plt.ylabel(r'$\Im (\vec V_{\mu})$ [V]')
plt.legend()
plt.show()
