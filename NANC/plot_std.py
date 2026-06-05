import numpy as np
import matplotlib.pyplot as plt


plt.rc('font', family='serif')
plt.rc('font', size=22)
plt.rc('axes', labelsize=26)
plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)
plt.rc('legend', fontsize=22)
plt.rc('figure', titlesize=22)


num_cavities = 8
cav_names = [f"Cav{i+1}" for i in range(num_cavities)]

data_std = np.loadtxt("std.csv", delimiter=",")
data_peak = np.loadtxt("peak.csv", delimiter=",")

x_axis = np.linspace(1,6,6)
fig, ax1 = plt.subplots(figsize=(22, 9))
colors = plt.cm.tab10(np.linspace(0, 1, num_cavities))
legend_handles = []
for i, name in enumerate(cav_names):
    (line_peak,) = ax1.plot(
        x_axis,
        data_peak[:,i],
        color=colors[i],
        linestyle="-",
        marker="o",
        ms=12,
        lw=4,
        label=name,
    )

    ax1.plot(
        x_axis,
        data_std[:,i]*3,
        color=colors[i],
        linestyle="--",
        marker="^",
        ms=12,
        lw=4
    )

    legend_handles.append(line_peak)

ax1.set_xlabel("Capture Run #")
ax1.set_ylabel("Cavity Metrics [Hz]")
cavity_legend = ax1.legend(
    handles=legend_handles,
    labels=cav_names,
    loc="upper left"
)
#ax1.add_artist(cavity_legend)
from matplotlib.lines import Line2D

style_handles = [
    Line2D(
        [0],
        [0],
        color="gray",
        linestyle="-",
        marker="o",
        ms=12,
        lw=4,
        label="Detuning",
    ),
    Line2D(
        [0],
        [0],
        color="gray",
        linestyle="--",
        marker="^",
        ms=12,
        lw=4,
        label="Std Dev * 3",
    ),
]

all_handles = legend_handles + style_handles
plt.legend(
    handles=all_handles,
    loc="upper right",
    ncol=1
)
plt.show()