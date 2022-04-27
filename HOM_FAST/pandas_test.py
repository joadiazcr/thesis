# I have to test how to load a csv file into pandas frame
# and how to normally plot one column vs other column,
# after loading the data and filtering the data I care about

import pandas as pd
import matplotlib.pyplot as plt

# Load
data = pd.read_csv("HOM_FAST/shift3_20201120/cmhoms_peaks_pd_combined.csv")
print(data.head())

# Filter
cav1 = data[data['cav'] == '1'][data['loc'] == 'upstr']
print(cav1.shape)
print(cav1.head(20))

# data type
print(data.dtypes)

cav1.plot()
plt.show()
