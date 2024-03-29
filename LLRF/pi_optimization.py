import numpy as np
import sys
import os
import argparse
from scipy.optimize import minimize
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from cavity_step_response import cavity_step as cav_stp
from pi_gain_analysis import error, rmse


# Example from:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
def test_function():
    print('\nSLSQP')

    def fun(x):
        print(x)
        y = (x[0] - 1)**2 + (x[1] - 2.5)**2
        return y
    x0 = [2, 0]
    cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
            {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
            {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
    bnds = ((0, None), (0, None))
    res = minimize(fun, x0, method='SLSQP', bounds=bnds,
                   constraints=cons, options={'disp': True})
    print('Solution =', res.x)


def cavity_step_rmse(s_gbw):
    start = datetime.now()
    # Some settings for the simulation
    tf = 0.03  # Total simulation time
    nom_grad = 1.6301e7  # V/m. Nominal gradient of LCLS-II cavity
    beam = False
    detuning = True
    noise = True
    feedforward = True
    noise_psd = -138
    beam_start = 0.015
    beam_end = 0.04
    detuning_start = 0.0
    detuning_end = tf

    # 1. Simulate cavity response with given gains
    args = [tf, nom_grad, feedforward, noise, noise_psd, beam, beam_start,
            beam_end, detuning, detuning_start, detuning_end]
    t, cav_v, sp, fwd, Kp_a, Ki_a, e, ie = cav_stp(args,
                                                   stable_gbw=s_gbw)

    # 2. Calculate error
    err = error(sp, np.abs(cav_v))

    # 3. Calculate RMSE
    rms_err = rmse(err)
    end = datetime.now()
    exec_time = end - start
    print(str(s_gbw), '\t\t{:.6f}'.format(rms_err), '\t', exec_time)
    return rms_err


if __name__ == "__main__":
    des = 'Optimize proportional and integral gains of the LLRF\
           controller by minimizing the rmse of the cavity amplitude'
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('-t', "--test", action="store_true",
                        help='test cavity step rmse calculation function')

    args = parser.parse_args()

    s_gbw = 40000  # Initial guess

    if args.test:
        print('\nTesting cavity_step_rmse...')
        rms_err = cavity_step_rmse(s_gbw)
        print('For 0-dBcrossing', s_gbw, 'the rmse is', rms_err)

    else:
        print('\nNelder-Mead')
        print("0-dB crossing\t\trmse\t\tExec Time")
        res = minimize(cavity_step_rmse, s_gbw, method='SLSQP',
                       tol=1e-1, options={'disp': True, 'maxiter': 1})
        print('Solution =', res.x)
