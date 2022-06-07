import matplotlib.pyplot as plt
import pandas as pd

bunch_rate = 3e6  # Hz
bunch_T = 1/bunch_rate


def bunch2us(x):
    return x * bunch_T * 1e6


def us2bunch(x):
    return x / (bunch_T * 1e6)


df = pd.read_pickle('HOM_FAST/bpm_data_ex.plk')

df = df[df['connection'] == 'Upstream']
df = df.reset_index(drop=True)

BPMs = ['B418PH', 'B418PV', 'B440PH', 'B440PV',
        'B441PH', 'B441PV', 'B444PH', 'B444PV']
num_BPMs = len(BPMs)
num_rows = 2
num_cols = 4

# Single BPM plot
fig, axs = plt.subplots(figsize=(7, 6))
bpm = 5
bpm_name = BPMs[bpm]
for v125 in range(0, len(df['V125'])):
    bpm_mean = df[bpm_name + '_data'][v125]
    bpm_mean_mean = df[bpm_name + '_mean'][v125]
    label = (str(df['V125'][v125]) + ' A')
    axs.plot(bpm_mean - bpm_mean_mean, label=label)
axs.set_xlabel('Bunch Number', fontsize=16, fontweight='bold')
axs.set_ylabel('Position ' + r'$\mathbf{[\mu m]}$', fontsize=16,
               fontweight='bold')
secax = axs.secondary_xaxis('top', functions=(bunch2us, us2bunch))
secax.set_xlabel('Time ' + r'$\mathbf{[\mu s]}$', fontsize=16,
                 fontweight='bold')
secax.set_xticklabels(['0', '0', '2', '4', '6', '8', '10', '12', '14', '16'],
                      fontsize=14, fontweight='bold')

legend_properties = {'weight': 'bold', 'size': 14}
plt.legend(loc='upper right', prop=legend_properties)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.grid()
plt.show()
exit()

fig, axs = plt.subplots(num_rows, num_cols, sharey=True, sharex=True)
title = ('HEBPMS Upstream 250pC/b 50b H125@reference')
fig.suptitle(title, fontsize=20)  # Hardcoded title
for bpm in range(0, num_BPMs):
    bpm_name = BPMs[bpm]
    title = bpm_name
    for v125 in range(0, len(df['V125'])):
        bpm_mean = df[bpm_name + '_data'][v125]
        bpm_mean_mean = df[bpm_name + '_mean'][v125]
        label = (str(df['V125'][v125]) + ' A')
        axs[int(bpm / num_cols), bpm % num_cols].plot(bpm_mean - bpm_mean_mean,
                                                      label=label)
    axs[int(bpm / num_cols), bpm % num_cols].grid(True)
    title = (bpm_name)
    axs[int(bpm / num_cols), bpm % num_cols].set_title(title, fontsize=18)
    axs[int(bpm / num_cols), bpm % num_cols].tick_params(axis='both',
                                                         which='major',
                                                         labelsize=16)
    if int(bpm / 4) == 1:
        axs[int(bpm / 4), bpm % 4].set_xlabel('Bunch Number', fontsize=18)
    if bpm % 4 == 0:
        axs[int(bpm / 4), bpm % 4].set_ylabel('Position (um)', fontsize=18)

    plt.legend(loc='upper right', fontsize=14)

plt.show()
