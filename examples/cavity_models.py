import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import signal

# Transfer funciton, state space and differential equation models for th cavity equation 3.13

# Differential equation model
def cavity_de(z, t, RoverQ, Qg, Q0, Qprobe, bw, Kg_r, Kg_i, Ib,
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


# Transfer function model
def cavity_tf(RoverQ, Qg, Q0, Qprobe, bw, Kg_r, Kg_i, Ib,
           foffset):

        Rg = RoverQ * Qg
        Kdrive = 2 * np.sqrt(Rg)
        Ql = 1.0 / (1.0/Qg + 1.0/Q0 + 1.0/Qprobe)
        K_beam = RoverQ * Ql
        w_d = 2 * np.pi * foffset

        a = Kdrive * bw
        b = K_beam * bw
        c = bw

	num = a/c
	den = [1.0/c,1.0]
	sys = signal.TransferFunction(num,den)
	t, y = signal.step(sys) 

        return t, y


# State Space Model
def cavity_ss(RoverQ, Qg, Q0, Qprobe, bw, Kg_r, Kg_i, Ib,
           foffset):

        Rg = RoverQ * Qg
        Kdrive = 2 * np.sqrt(Rg)
        Ql = 1.0 / (1.0/Qg + 1.0/Q0 + 1.0/Qprobe)
        K_beam = RoverQ * Ql
        w_d = 2 * np.pi * foffset

        a = Kdrive * bw
        b = K_beam * bw
        c = bw

	A = -c
	B = a
	C = 1.0
	D = 0

	sys = signal.StateSpace(A,B,C,D)
	t,y = signal.step(sys)

        return t, y


# Pi mode cavity parameters
bw = 104.0  # where is this number coming from? Cavity bw 16Hz
RoverQ_pi = 1036.0
Qg_pi = 4.0 * 10**7
Q0_pi = 2.7 * 10**10
Qprobe_pi = 2.0 * 10**9

# Cavity conditions
foffset = 0.0
Kg = 1.0
Ib = 0.0

# Simulation time variables
tf = 0.05  # Total simulation time
nsteps = 500  # number of time steps
delta_t = tf/(nsteps-1)
ts = np.linspace(0, tf, nsteps)  # linearly spaced time vector

args_pi = (RoverQ_pi, Qg_pi, Q0_pi, Qprobe_pi, bw, Kg, 0.0, Ib, foffset)
z0 = np.zeros(3)
t_tf, y_tf = cavity_tf(RoverQ_pi, Qg_pi, Q0_pi, Qprobe_pi, bw, Kg, 0.0, Ib, foffset)
t_ss, y_ss = cavity_ss(RoverQ_pi, Qg_pi, Q0_pi, Qprobe_pi, bw, Kg, 0.0, Ib, foffset)
y_de = odeint(cavity_de, z0, ts, args_pi)

print y_de

plt.figure(1)
plt.plot(t_ss,y_ss,'g:',linewidth=2,label='State Space')
plt.plot(t_tf, y_tf,'b--',linewidth=3,label='Transfer Fcn')
plt.plot(ts, y_de[:, 0],'r-',linewidth=1,label='ODE Integrator')
plt.xlabel('Time')
plt.ylabel('Response (y)')
plt.legend(loc='best')
plt.show()
