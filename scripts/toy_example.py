# toy example
import multiprocessing as mp
import numpy as np
import time as tm

def func(a):
    a = a*2
    tm.sleep(1)
    return a

num_processes = mp.cpu_count()

for count in range(30):
    k = list(np.arange(1000))
    Pool = mp.Pool(processes = num_processes)
    result = list(Pool.map(func, k, 5))
    Pool.close()

    print(result[0])