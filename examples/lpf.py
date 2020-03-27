import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Simulate taup * dy/dt = -y + K*u
Kp = 0.6
f = 1e6
taup = 1.0/(f*2.0*np.pi)

# Simulation time
Tmax = 1e-6
Tstep = 1e-9
trang = np.arange(0, Tmax, Tstep)
nt = len(trang)

# (1) Transfer Function
num = [Kp]
den = [taup,1]
sys1 = signal.TransferFunction(num,den)
t1,y1 = signal.step(sys1, T=trang)

# (2) State Space
A = -1.0/taup
B = Kp/taup
C = 1.0
D = 0.0
sys2 = signal.StateSpace(A,B,C,D)
t2,y2 = signal.step(sys2, T=trang)

# (3) ODE Integrator
def model3(y,t):
    u = 1
    return (-y + Kp * u)/taup
y3 = odeint(model3, 0, trang)

# (3.a) Other way to implement ODE
y4 =  np.zeros(nt)
for i in xrange(1,nt):
    y_last = y4[i-1]
    y_out = odeint(model3, y_last, [0,Tstep])
    y4[i] = y_out[-1]

j = odeint(model3, 0.42094026551528124, [0,Tstep])
print j

plt.figure(1)
plt.plot(trang,y1,'b--',linewidth=3,label='Transfer Fcn')
plt.plot(trang,y2,'g:',linewidth=2,label='State Space')
plt.plot(trang,y3,'r-',linewidth=1,label='ODE Integrator')
plt.plot(trang,y4,'p-',linewidth=1,label='ODE.a Integrator')
plt.hlines(Kp*(1-np.exp(-1)), 0, trang[-1])
plt.hlines(Kp, 0, trang[-1])
plt.vlines(taup, 0, Kp)
plt.xlabel('Time')
plt.ylabel('Response (y)')
plt.legend(loc='best')
plt.show()
