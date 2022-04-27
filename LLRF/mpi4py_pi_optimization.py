from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
from scipy.optimize import minimize
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from cavity_step_response import cavity_step
from pi_gain_analysis import error, rmse


def log(logfile, line, stdout=False):
    logfile.write(line+'\n')
    if stdout:
        print(line)
        sys.stdout.flush()


def test_func(x):
    np.set_printoptions(precision=4)
    start = datetime.now()
    y = np.sin(x) + x**2/10000000
    end = datetime.now()
    exec_time = end - start
    log(logfile, '{:.6f}'.format(x) + '\t' + y + '\t' + str(exec_time))
    return y

def cavity_step_rmse(s_gbw):
    start = datetime.now()
    # Some settings for the simulation
    tf = 0.03 # Total simulation time
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


if __name__ == "__main__":
    des = "Optimize proportional and integral gains of the LLRF\
           controller by minimizing the rmse of the cavity amplitude.\
           This script uses MPI4PY to run in parallel by using multiple\
           cores at the same time.\
           Run as: mpirun -np <#> python3 test_mpi4py.py ..."
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('-d', '--dir', dest='log_root', default="opt_results_",
                        help='Log/data directory prefix (can include path)')

    args = parser.parse_args()


    start_time = datetime.now()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create log directory
    data_dir = ''
    if rank == 0:
        data_dir = start_time.strftime(args.log_root + '%Y%m%d_%H%M%S')
        os.mkdir(data_dir)
    comm.Barrier()
    data_dir = comm.bcast(data_dir, root=0)

    opt_methods = ['SLSQP', 'L-BFGS-B', 'TNC', 'Trust-constr']

    # Create log file
    logfile = open(data_dir + '/' + opt_methods[rank], "a")
    log(logfile, 'Optimization method:' + opt_methods[rank], stdout=False)

    s_gbw = 40000 # Initial guess
    method = opt_methods[rank]
    bnds = [(39000, 41000)]

    log(logfile, "0-dB crossing\trmse\t\tExec Time")
    res = minimize(cavity_step_rmse, s_gbw, method=method, bounds=bnds, 
                   tol=1e-1, options={'disp': False, 'maxiter': 3})
    log(logfile, 'Solution =' + str(res.x), stdout=True)



