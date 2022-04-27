import matplotlib.pyplot as plt
import argparse


def load_data(file):
    gain_s = []
    rmse_s = []
    results = open(file, 'r')
    for line in results:
        if 'Optimization method' in line:
            method = line.split(':')[1]
            print('The optimization method is ', method)
        elif 'Solution' in line:
            opt_gain = line.split('[')[1]
            opt_gain = opt_gain.split(']')[0]
        elif 'Solution' not in line and '0-dB' not in line:
            gain_s.append(float(line.split()[0]))
            rmse_s.append(float(line.split()[1]))
    print(opt_gain)
    return method, gain_s, rmse_s


def plot_opt_results(method, gain_s, rmse_s):
    print('Plotting results...')
    plt.title(method)
    plt.plot(gain_s, rmse_s, '*')
    plt.xlabel('0-dB crossing [Hz]')
    plt.ylabel('RMSE [V]')
    plt.show()


if __name__ == "__main__":
    des = 'Plot results of optimization'
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('-f', '--file', dest='log_file', required=True,
                        help='File with opt results to be plotted')

    args = parser.parse_args()

    method, gain_s, rmse_s = load_data(args.log_file)
    plot_opt_results(method, gain_s, rmse_s)
