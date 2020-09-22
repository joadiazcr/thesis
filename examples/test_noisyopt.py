import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der
from datetime import datetime
import time
import argparse
import sys

import numpy as np
from noisyopt import minimizeCompass, minimizeSPSA

import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from cavity_step_response import cavity_step
from pi_gain_analysis import error, rmse


class Opt_Results:
    def __init__(self, res, method):
        self.method = method
        self.res = res
        self.opt_y = self.res.fun
        self.opt_x = self.res.x
        print(self.res.items())


    def plot_opt(self, x_s, y_s, x, y):
        plt.title(self.method)
        plt.plot(x, y, '--', label='Function')
        plt.plot(x_s, y_s, 'o', label='Function evaluations [' + str(len(x_s)) + ']')
        plt.plot(self.opt_x, self.opt_y, '*', label='Optimal (' + str(self.opt_x) + ',' + str(self.opt_y) + ')')
        plt.legend()
        plt.show()


def obj(x, save=True):
    y = (x**2).sum() + 0.1*np.random.randn()
    if save:
        log(logfile, str(x[0]) + '\t' + str(y), stdout=True)
    return y


def cavity_step_rmse(s_gbw):
    start = datetime.now()
    # Some settings for the simulation
    tf = args.time # Total simulation time in seconds
    nom_grad = 1.6301e7 # V/m. Nominal gradient of LCLS-II cavity
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
    t, cav_v, sp, fwd, Kp_a, Ki_a, e, ie = cavity_step(tf, nom_grad, feedforward, noise, noise_psd,\
                                                             beam, beam_start, beam_end, detuning,\
                                                             detuning_start, detuning_end, stable_gbw=s_gbw)

    # 2. Calculate error
    err = error(sp, np.abs(cav_v))

    # 3. Calculate RMSE
    rms_err = rmse(err)
    #rms_err = 100.0 + np.random.rand(1,1)
    end = datetime.now()
    exec_time = end - start
    log(logfile, '{:.6f}'.format(s_gbw[0]) + '\t{:.6f}'.format(rms_err) + '\t' + str(exec_time),
        stdout=True)
    return rms_err



def log(logfile, line, stdout=False):
    logfile.write(line+'\n')
    if stdout:
        print(line)
        sys.stdout.flush()


if __name__ == "__main__":
    des = 'script to test optimization algorithms'
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('-obj', "--obj", action="store_true",
                        help='Test function obj')
    parser.add_argument('-t', "--time", dest="time", default=0.02, type=float,
                        help='Cavity simulation time')

    args = parser.parse_args()

    start_time = datetime.now()

    if args.obj:
        print('Optimizing function obj...')

        # Create log file
        logfile_name = start_time.strftime('test_noisyopt_results' + '%Y%m%d_%H%M%S')
        print("Log file: %s" % logfile_name)
        logfile = open(logfile_name, "w")
        log(logfile, 'Optimization method: noisyopt', stdout=False)
        log(logfile, "0-dB crossing\trmse\t\tExec Time")

        x = np.linspace(-1, 1, 100)
        y = []
        for i in x:
            y.append(obj(i, False))

        bnds = [(-1, 1)]
        x0 = [1.0]
        #res = minimizeCompass(obj, x0=[1.0], deltatol=0.1, paired=False, funcNinit=10, alpha=0.01)
        bnds = [(10000, 70000)]
        x0 = [10000]
        res = minimizeCompass(cavity_step_rmse, x0, bounds=bnds, paired=False, funcNinit=3, deltatol=20, deltainit=60000)

        # Load Func evaluations
        logfile = open(logfile_name, "r")
        x_s = []
        y_s = []
        for line in logfile:
            x_s.append(float(line.split()[0]))
            y_s.append(float(line.split()[1]))

        min_scalar_results = Opt_Results(res, 'Noisyopt')
        min_scalar_results.plot_opt(x_s, y_s, x, y)
