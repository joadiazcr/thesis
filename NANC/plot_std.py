import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('std.csv', header=None)

# Plot all columns together
df.plot(figsize=(10, 6))
plt.title("All Columns Overview")
plt.ylabel("Value")
plt.xlabel("Index")
plt.grid(True)
plt.show()