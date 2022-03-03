import numpy as np
import scipy
from matplotlib import pyplot as plt
from NI_simulation.modules import ni_sim
from scipy import interpolate
import sys
import pandas as pd
from scipy import signal
from tqdm import tqdm

# create sources
sources = ni_sim.source_distribution2D().distant_uniform(5000, 30000, 4000, label='gauss')

env = ni_sim.environment(sources, time_length=600, frequencies = np.linspace(1,20,20))

print('Try 1')
_ = env.get_correlations(verbose=True, correlation_type='all', chunksize=10)

print('Try 2')
_ = env.get_correlations(verbose=True, correlation_type='all', chunksize=10)