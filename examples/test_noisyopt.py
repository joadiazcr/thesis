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


def quad_func(x, *args):
    y = (x**2).sum() + 0.2*np.random.randn()
    save = args[0]
    if save:
        log(logfile, str(x[0]) + '\t' + str(y), stdout=True)
    return y


def cavity_step_rmse(s_gbw, *args):
    start = datetime.now()
    # Some settings for the simulation
    tf = args[0] # Total simulation time in seconds
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


def load_data(file):
    print('Loading data from %s' % file)
    gain_s = []
    rmse_s = []
    opt_gain = []
    method = ''
    time_s = []
    results = open(file, 'r')
    for line in results:
        if 'Optimization method' in line:
            method = line.split(':')[1]
            print('The optimization method is ', method)
        elif 'Solution' in line:
            opt_gain = line.split('[')[1]
            opt_gain = opt_gain.split(']')[0]
        elif 'Solution' not in line and  '0-dB' not in line\
                and '=' not in line and 'rank' not in line\
                and 'Results' not in line:
            gain_s.append(float(line.split()[0]))
            rmse_s.append(float(line.split()[1]))
            try:
                time_s.append(line.split()[2])
            except:
                pass

    return method, gain_s, rmse_s, time_s


if __name__ == "__main__":
    des = 'script to test optimization algorithms'
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('-f', "--func", choices=['quad_func','cavity_step_rmse'],
                        help='Function to optimize: quad_func or cavity_step_rmse')
    parser.add_argument('-t', "--time", dest="time", default=0.02, type=float,
                        help='Cavity simulation time')
    parser.add_argument('-fni', "--fni", dest="funcNinit", default=3, type=int,
                        help='Parameter funcNinit for minimizeCompass function')
    parser.add_argument('-di', "--di", dest="deltainit", default=60000, type=int,
                        help='Parameter deltainit for minimizeCompass function')
    parser.add_argument('-feps', "--feps", dest="feps", default=30, type=float,
                        help='Parameter feps for minimizeCompass function')

    args = parser.parse_args()

    start_time = datetime.now()

    funcNinit = args.funcNinit
    deltainit = args.deltainit
    feps = args.feps

    x = []
    y = []

    if args.func:
        print('Optimizing function obj...')

        # Create log file
        logfile_name = start_time.strftime('test_noisyopt_results' + '%Y%m%d_%H%M%S')
        print("Log file: %s" % logfile_name)
        logfile = open(logfile_name, "w")
        log(logfile, 'Optimization method: noisyopt', stdout=False)
        log(logfile, "0-dB crossing\trmse\t\tExec Time")

        if args.func == 'quad_func':
            bnds = [(-1, 1)]
            x0 = [1.0]
            save = True
            res = minimizeCompass(quad_func, x0, args=(save, save), bounds = bnds, deltainit=deltainit, funcNinit=funcNinit, paired=False, feps=feps, disp=False)

            x = np.linspace(-1, 1, 100)
            for i in x:
                y.append(quad_func(i, False))

        elif args.func == 'cavity_step_rmse':
            bnds = [(10000, 70000)]
            x0 = [10000]
            res = minimizeCompass(cavity_step_rmse, x0, args=(args.time, args.time), bounds=bnds, paired=False, funcNinit=funcNinit, deltainit=deltainit, feps=feps, disp=False)

        log(logfile, str(res.items()), stdout=True)
        logfile.close()
        method, x_s, y_s, tme_s = load_data(logfile_name)

        min_scalar_results = Opt_Results(res, 'Noisyopt')
        min_scalar_results.plot_opt(x_s, y_s, x, y)
