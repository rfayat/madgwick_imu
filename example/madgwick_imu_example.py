"""
Run Madgwick algorithm on an example dataset.

Author: Romain Fayat

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from madgwick import Madgwick
from pathlib import Path
from cycler import cycler

# Paramaeters
FREQUENCY = 300.  # Sampling rate (in Herz)

# Data Loading
data_path = Path(__file__).parent / "imu_data.csv"
data = pd.read_csv(data_path, index_col=0)
acc = data[["ax", "ay", "az"]].values  # Acceleration, in
gyr = np.radians(data[["gx", "gy", "gz"]].values)  # Angular speed, in radin

# Create the filter and compute the estimate of gravity
# in the IMU reference frame
mf =  Madgwick(acc=acc, gyr=gyr, frequency=FREQUENCY, gain=.1)
gravity_estimate = mf.gravity_estimate()

# Plot the result and save the figure
rgb_cycler = cycler(color=["#D55E00", "#009E73", "#0072B2"])
fig, ax = plt.subplots(figsize=(5, 3))
ax.set_prop_cycle(rgb_cycler)
ax.plot(data.time, acc, alpha=.5)
ax.plot(data.time, gravity_estimate, linewidth=1.5, label=["x", "y", "z"])
ax.legend(loc="lower left")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Acceleration [G]")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "script_out.png")
