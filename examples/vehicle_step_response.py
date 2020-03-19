import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#define vehicle model.
def vehicle(v, t, u, load):
	# inputs
	# v = vehicle velocity (m/s)
	# t = time (sec)
	# u = gas pedal position (-50% to 100%)
	# load = passenger load + cargo (kg)
	Cd = 0.24	# drag coefficient
	rho = 1.225	# air density (kg/m^3)
	A = 5.0		# cross-sectional area (m^2)
	Fp = 30		# thrust parameter (N/%pedal)
	m = 500		# vehicle mass (kg)
	# calculate derivative of the velocity. Differential equation
	dv_dt = (1.0/(m + load)) * (Fp*u - 0.5 * rho* Cd * A * v**2)
	return dv_dt

tf = 100.0 	# simulation time
nsteps = 101	# number of time steps
delta_t = tf/(nsteps-1)
ts = np.linspace(0, tf, nsteps) # linearly spaced time vector

# simulate step test operation
step = np.zeros(nsteps)
step[11:] = 25.0 # step up pedal position. u variable
load = 200.0 # passenger load + cargo (kg)
v0 = 0.0 # velocity initial condition
vs = np.zeros(nsteps) # for storing the results

# set point
sp = 25.0
sp_store = np.zeros(nsteps)
sp_store[0:] = sp # step up pedal position. u variable

# PI controller
#Kp = 1.0 / 1.2
#Ki = Kp / 20.0
Kp = 3.0
Ki = 0.1
sum_init = 0.0# Integral(sumation) initial value
e = np.zeros(nsteps) # error initialization
ie = np.zeros(nsteps) # integral of the error initialization

# simulate with ODEINT
for i in range(nsteps-1):
        error = sp - v0
        e[i+1] = error
        sum_init = sum_init + error * delta_t
	
	if Kp == 0.0 and Ki == 0.0:
		u = step[i]
	else:
	        u = Kp * error + Ki * sum_init
	# clip inputs to -50% to 100%
	if u >= 100.0:
		u = 100.0
		sum_init = sum_init - error *delta_t
        if u <= -50.0:
                u = -50.0
                sum_init = sum_init - error *delta_t
	ie[i+1] = sum_init
	if Kp != 0.0 and Ki != 0.0:
		step[i+1] = u
	v = odeint(vehicle, v0, [0, delta_t], args = (u, load))
	v0 = v[-1] # take the last value of v
	vs[i+1] = v0

# plot results
plt.figure()
plt.subplot(2, 2 ,1)
plt.plot(ts, vs, 'b-', linewidth = 3, label = 'Velocity')
plt.plot(ts, sp_store, 'k--', linewidth = 2, label = 'Set point')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.subplot(2, 2 ,2)
plt.plot(ts, step, 'r--', linewidth = 3, label = 'Gas Pedal (%)')
plt.ylabel('Gas Pedal')
plt.legend()
plt.subplot(2, 2 ,3)
plt.plot(ts, e, 'b-', linewidth = 3, label = 'Error (SP_PV)')
plt.xlabel('Time (sec)')
plt.legend()
plt.subplot(2, 2 ,4)
plt.plot(ts, ie, 'b-', linewidth = 3, label = 'Integral of error')
plt.xlabel('Time (sec)')
plt.legend()
plt.show()
