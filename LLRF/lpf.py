import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def odeintz(func, z0, t, **kwargs):
    """An odeint-like function for complex valued differential equations."""

    # Disallow Jacobian-related arguments.
    _unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]
    if len(bad_args) > 0:
        raise ValueError("The odeint argument %r is not supported by "
                         "odeintz." % (bad_args[0],))

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        dzdt = func(z, t, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = odeint(realfunc, z0.view(np.float64), t, **kwargs)

    if kwargs.get('full_output', False):
        z = result[0].view(np.complex128)
        infodict = result[1]
        return z, infodict
    else:
        z = result.view(np.complex128)
    return z


# Simulate taup * dy/dt = -y + K*u
Kp = 0.6
f = 1e6
taup = 1.0/(f*2.0*np.pi)

# Simulation time
Tmax = 1e-6
Tstep = 1e-7
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


# (4) ODE Integrator with complex numbers
def model4_complex(y0, t):
    y0_r = np.real(y0)
    y0_i = np.imag(y0)
    u_r = 0.6
    u_i = 0.0
    dydt_r = -f*2.0*np.pi * y0_r + f*2.0*np.pi * u_r
    dydt_i = -f*2.0*np.pi * y0_i + f*2.0*np.pi * u_i
    dydt = dydt_r + dydt_i * 1.0j
    return dydt
y4 = odeintz(model4_complex, 0+1.0j, trang)


# (4.a) ODE Integrator with complex numbers and no ODEINTZ
def model4_complex_a(y0, t):
    y0_r, y0_i = y0
    u_r = 0.6
    u_i = 0.0
    dydt_r = -f*2.0*np.pi * y0_r + f*2.0*np.pi * u_r
    dydt_i = -f*2.0*np.pi * y0_i + f*2.0*np.pi * u_i
    return dydt_r, dydt_i
y4a =  np.zeros(nt, dtype='complex')
for i in range(1,nt):
    y_last_r = np.real(y4a[i-1])
    y_last_i = np.imag(y4a[i-1])
    y_last = [y_last_r, y_last_i]
    y = odeint(model4_complex_a, y_last, [0,Tstep])
    y4a[i] = (y[-1, 0] + 1j*y[-1, 1])


# (3.a) Other way to implement ODE
y3a =  np.zeros(nt)
for i in range(1,nt):
    y_last = y3a[i-1]
    y_out = odeint(model3, y_last, [0,Tstep])
    y3a[i] = y_out[-1]


plt.figure(1)
plt.plot(trang,y1,'b--',linewidth=3,label='Transfer Fcn')
plt.plot(trang,y2,'g:',linewidth=2,label='State Space')
plt.plot(trang,y3,'r-',linewidth=1,label='ODE Integrator')
plt.plot(trang,y3a,'p-',linewidth=1,label='ODE.a Integrator')
plt.plot(trang,np.abs(y4),'p-',linewidth=1,label='ODEINTZ Integrator complex')
plt.plot(trang,np.abs(y4a),'*',linewidth=1,label='ODEINTZ Integrator complex no ODEINTz')


plt.hlines(Kp*(1-np.exp(-1)), 0, trang[-1])
plt.hlines(Kp, 0, trang[-1])
plt.vlines(taup, 0, Kp)
plt.xlabel('Time')
plt.ylabel('Response (y)')
plt.legend(loc='best')
plt.show()