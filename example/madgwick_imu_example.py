"""
Run Madgwick algorithm on an example dataset.

Author: Romain Fayat

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from madgwick.madgwick import computeGravity

sr = 300. # Hz
labels_acc = ["ax", "ay", "az"]
labels_gyr = ["gx", "gy", "gz"]
labels_gravity = ["ax_G", "ay_G", "az_G"]


def deg_to_rad(arr):
    "Convert angular data from degrees to radians."
    return arr * (2 * np.pi / 360)  # Rescaled data


# Load the example dataset
data_path = "example/IMU_data_sample.txt"
data = pd.read_csv(data_path, encoding="utf-8", sep="\s+")

# Add a column corresponding to the elapsed time
data["time"] = np.arange(len(data)) / sr
# Convert the gyr labels from degrees to radians
data[labels_gyr] = deg_to_rad(data[labels_gyr].values)

# Compute the gravity
acc = data[labels_acc].values
gyr = data[labels_gyr].values
gravity_estimate = pd.DataFrame(computeGravity(acc, gyr , sampleFreq=sr),
                                columns=labels_gravity)

data[labels_gravity] = gravity_estimate

# plot
start, end = 100, 120  # seconds
target = (data.time > start) & (data.time < end)
fig, ax = plt.subplots()
for i, orientation in enumerate(["x", "y", "z"]):
    ax.plot(data.time[target], data[f"a{orientation}"][target], c="rgb"[i],
            label=f"acc {orientation}")
    ax.plot(data.time[target], data[f"a{orientation}_G"][target], c="k",
            label=f"Gravity estimate {orientation}", linewidth=.5)
ax.legend(loc="lower right")
ax.set_xlim([start, end])
ax.set_title("Example of gravity estimate using Madgwick's algorithm")

fig.savefig("example/script_out.png")
