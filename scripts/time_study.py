# Add Code Directory (TEMPORY FIX)
import sys
sys.path.append('/home/jhrag/Code')

import numpy as np
from NI_simulation.modules import ni_sim
from scipy import signal
from matplotlib import pyplot as plt
from scipy.io import wavfile
import scipy.io
import datetime

num_sources = [15, 100, 300, 500, 700, 1000, 1500]
avg_times = [5400, 7200, 9000]

num_source_list = []
avg_time_list = []
time = []
for num_source in num_sources:
    for avg_time in avg_times:
        print(f'Simulating for {avg_time} seconds with {num_source} sources')
        starttime = datetime.datetime.now()
        source_distribution = ni_sim.source_distribution2D()
        source_distribution.endfire_circle(10,10000,num_source)
        sources = source_distribution.sources
        env = ni_sim.environment(sources)
        xA, xB = env.get_signals()
        endtime = datetime.datetime.now()
        delT = endtime-starttime
        num_source_list.append(num_source)
        avg_time_list.append(avg_time)
        time.append(delT.microseconds)

print('Number of Sources:')
for x in num_source_list:
    print(x)

print('Average Times:')
for x in avg_time_list:
    print(x)

print('Computation Times (μs):')
for x in time:
    print(x)
