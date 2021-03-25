import numpy as np
import scipy
from matplotlib import pyplot as plt
from NI_simulation.modules import ni_sim
from scipy import interpolate
import sys

# create sources
sources = ni_sim.source_distribution2D().endfire_circle(10, 10000, 200)

env = ni_sim.environment(sources, time_length=600)

xA, xB = env.get_signals()

R = np.correlate(xA, xB, mode='full')
tx = np.linspace(-env.time_length, env.time_length, len(xA))

plt.plot(tx,xA)
plt.show()

sys.exit()