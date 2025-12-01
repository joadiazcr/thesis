import numpy as np
import glob

# Get all txt files in folder
files = sorted(glob.glob("/home/dejorge/Documents/NANC/shift_20251106/res_CM19_cav2_*"))

# Load each file as a column
arrays = [np.loadtxt(f) for f in files]

# Stack horizontally (columns)
data = np.column_stack(arrays)

np.savetxt("/home/dejorge/Documents/NANC/shift_20251106/combined_cm19_cav2", data, fmt="%.6f")
