import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der
from datetime import datetime
import time
import argparse
import sys


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
        plt.plot(self.opt_x, self.opt_y, '*', label='Optimal')
        plt.legend()
        plt.show()


def f1(x, save=True):
    y = (x - 2) * x * (x + 2)**2
    if save:
        log(logfile, str(x[0]) + '\t' + str(y[0]), stdout=False)
    return y


def f2(x):
    return np.cos(14.5 * x - 0.3) + (x + 0.2) * x


def log(logfile, line, stdout=False):
    logfile.write(line+'\n')
    if stdout:
        print(line)
        sys.stdout.flush()

if __name__ == "__main__":
    des = 'script to test optimization algorithms'
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('-f1', "--f1", action="store_true",
                        help='Test function f1')
    parser.add_argument('-f2', "--f2", action='store_true',
                        help='Test function f2')
    parser.add_argument('-nm', "--neldermead", action='store_true',
                        help='Test optimization function Nelder Mead')
    parser.add_argument('-m', "--method", dest="method", default='L-BFGS-B',
                        help='Method for the optimization function')

    args = parser.parse_args()

    if args.f1:
        print('Testing f1...')
        x = np.linspace(-3, 3, 1000)
        y = f1(x)
        plt.plot(x,y)
        plt.show()

    if args.f2:
        print('Testing f2...')
        x = np.linspace(-3, 3, 1000)
        y = f2(x)
        plt.plot(x,y)
        plt.show()

    if args.neldermead:
        print('\nTesting Nelder-Mead optimization method...')

        # Create log file
        logfile_name = 'test_opt_algorithms_results'
        logfile = open(logfile_name, "w")

        n = 1000
        x = np.linspace(-3, 3, n)
        y = []
        for i in x:
            y.append(f1(i, False))

        method = args.method
        x0 = 3.0
        res = minimize(f1, x0, method=method, 
               tol=1e-1, options={'disp': True})

        # Load Func evaluations
        logfile = open(logfile_name, "r")
        x_s = []
        y_s = []
        for line in logfile:
            x_s.append(float(line.split()[0]))
            y_s.append(float(line.split()[1]))

        min_scalar_results = Opt_Results(res, method)
        min_scalar_results.plot_opt(x_s, y_s, x, y)
        exit()

            

print("\nRosen function for optimization performance test")
x = 0.1 * np.arange(10)
x = [3,3]
y = rosen(x)
print("Rosen of", x, "=", y)

x0 = [3] # Initial Guess

print('\nNelder-Mead')
res = minimize(rosen, x0, method='Nelder-Mead', 
               tol=1e-6, options={'disp': True})
print('Solution =', res.x)

exit()



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
res = minimize(fun, x0, method='SLSQP', bounds=bnds, tol=0.01,
               constraints=cons, options={'disp': True})
print('Solution =', res.x)
print('Solution =' + str(res.x))
print('Termination =' + str(res.message))
print('Evaluations of the function =' + str(res.nfev))
print('Iterations =' + str(res.nit))
print('Maximum constraint violation =' + str(res.maxcv))

exit()


def x_sqr(x):
        y = np.sin(x) + x**2/10000000
        print('x=', x, 'y=', y)
        return y

x = np.linspace(39000, 40000, 10000)
y = x_sqr(x)
plt.plot(x,y)
plt.show()

x0 = -100 # Initial Guess

method = 'SLSQP'
print('\n', method)
bnds = [(39000, 40000)]
start = datetime.now()
res = minimize(x_sqr, x0, method=method, bounds=bnds, 
               tol=1e-3, options={'disp': True, 'maxiter': 5})
end = datetime.now()
total_time = end - start
print('Solution =', res.x)
print('Execution time was ', total_time)
exit()

print('\nSLSQP')
bnds = [(-10, 10)]
start = datetime.now()
res = minimize(x_sqr, x0, method='SLSQP', bounds=bnds, 
               options={'disp': True})
end = datetime.now()
total_time = end - start
print('Solution =', res.x)
print('Execution time was ', total_time)
exit()


# Examples from:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html 

print("\nRosen function for optimization performance test")
x = 0.1 * np.arange(10)
x = [2,-1]
y = rosen(x)
print("Rosen of", x, "=", y)

x0 = [2, -1] # Initial Guess

print('\nNelder-Mead')
res = minimize(rosen, x0, method='Nelder-Mead', 
               tol=1e-6, options={'disp': True})
print('Solution =', res.x)

print('\nBFGS') # Gradient (Jacobian) needed
res = minimize(rosen, x0, method='BFGS', jac=rosen_der,
               options={'gtol': 1e-6, 'disp': True})
print('Solution =', res.x)

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