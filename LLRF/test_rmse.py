import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def rmse(error):
    n = len(error)
    mse = np.sum(np.abs(error**2)) / n
    rmse = np.sqrt(mse)
    return rmse


x = np.linspace(0, 2 * np.pi, 1000)
y = np.sin(x)

rmse = rmse(y)
std = np.std(y)

length = len(y)
mse_sk = mean_squared_error(np.zeros(length), y)
rmse_sk = np.sqrt(mse_sk)

print("rmse", rmse)
print("std", std)
print("rmse_sk", rmse_sk)

plt.plot(x, y)
plt.plot(x, np.ones(length)*rmse, '-', label='rmse')
plt.plot(x, np.ones(length)*std, 'o', label='std')
plt.plot(x, np.ones(length)*rmse_sk, '--', label='rmse_sk')
plt.legend()
plt.show()
