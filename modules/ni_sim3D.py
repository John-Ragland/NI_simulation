from typing import Counter
import numpy as np
from numpy.core.numeric import outer
import scipy
import pandas as pd
from scipy import interpolate
import multiprocessing as mp
from multiprocessing import Pool
import tqdm
from scipy.io import wavfile
import scipy.io
from scipy import signal
import xarray as xr

from matplotlib.lines import Line2D
from matplotlib import pyplot as plt


class environment:
    def __init__(self, sources, time_length=60):
        '''
        initialize environment variable
        Parameters
        ----------
        sources : pandas.DataFrame
            dataframe of source x and y cooridinates
        time_length : int
            length of simulation time in seconds. Default - 60
        '''
        # Define Environment Variabels
        self.Fs = 200
        f0 = 50 #Hz
        self.w0 = 2*np.pi*f0
        self.sigma = 1/(5*f0)

        self.c = 1500 # m/s
        
        self.time_length = time_length
        self.t = np.arange(0,self.time_length, 1/self.Fs)
        self.nodeA = (-1500, 0, 0)
        self.nodeB = (1500, 0, 0)
        self.depth = 1500 #m
        self.sources = sources

    def plot_env(self, xlim=None, ylim=None, type='both', ax=None):
        '''
        plot_env plot the environment that is being simulation

        Parameters
        ----------
        xlim : tuple
        ylim : tuple
        type : str
            "both" - creates subplot of side and top view
            "side" - plots only side view
        ax : matplotlib.ax
            if included, it just adds the plot to specified axis. Default is None
            (new axis is created). Currently only supported for 'side' type

        '''
        sources_x = self.sources.X
        sources_y = self.sources.Y
        sources_z = self.sources.Z

        if type == 'both':
            fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,5))

            ax1.plot(sources_x, sources_y, '.', color='green', markersize=15)
            ax1.plot(self.nodeA[0], self.nodeA[1], '.', color='red', markersize=15)
            ax1.plot(self.nodeB[0], self.nodeB[1], '.', color='red', markersize=15)
            ax1.set_title('Top View')
            ax1.set_xlabel('X (meters)')
            ax1.set_ylabel('Y (meters)')
            if xlim != None:
                ax1.set_xlim(xlim)
            if ylim != None:
                ax1.set_ylim(ylim)
            ax1.grid()

            ax2.plot(sources_x, sources_z, '.', color='green',markersize=15)
            ax2.plot(self.nodeA[0], self.nodeA[2], '.', color='red', markersize=15)
            ax2.plot(self.nodeB[0], self.nodeB[2], '.', color='red', markersize=15)
            ax2.set_title('Side View')
            ax2.set_xlabel('X (meters)')
            ax2.set_ylabel('Z (meters)')
            ax2.grid()

            if xlim != None:
                ax2.set_xlim(xlim)

            xlim = ax2.get_xlim()
            ax2.plot(xlim, [0,0], color='black')
            ax2.plot(xlim, [self.depth, self.depth], color='blue')
        
        if type == 'side':
            if ax == None:
                fig, ax = plt.subplots(1,1, figsize = (5,5))
            
            ax.plot(sources_x, sources_z, '.', color='green',markersize=15)
            ax.plot(self.nodeA[0], self.nodeA[2], '.', color='red', markersize=15)
            ax.plot(self.nodeB[0], self.nodeB[2], '.', color='red', markersize=15)
            ax.set_title('Side View')
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Z (meters)')
            ax.grid()

            if xlim != None:
                ax.set_xlim(xlim)

            xlim = ax.get_xlim()
            ax.plot(xlim, [0,0], color='black')
            ax.plot(xlim, [self.depth, self.depth], color='blue')

    def __get_radius(self, coord):
        '''
        gets radius to A and B for given coord

        Parameters
        ----------
        coord : tuple (length 2)
            (X,Y,Z) coordinate of source to calculate radius to nodes
        Returns
        -------
        rA : float
            radius from coord to A (in meters)
        rB : float
            radius from coord to B (in meters)
        '''
        rA = ((self.nodeA[0] - coord[0])**2 + (self.nodeA[1] - coord[1])**2 + (self.nodeA[2] - coord[2])**2)**0.5
        rB = ((self.nodeB[0] - coord[0])**2 + (self.nodeB[1] - coord[1])**2 + (self.nodeB[2] - coord[2])**2)**0.5
        return rA, rB

    def get_signals_1cpu(self, sources, rng=np.random.RandomState(0), num_reflections=10):
        '''
        get_signals calculated the time domain recieved signal at node A and B
            for a given source distribution
        
        Parameters
        ----------
        sources : pandas.DataFrame
            data frame containing the x and y information for every source
        
        Returns
        -------
        x_A : numpy array
            time series of recieved signal for node A (Sampled at self.Fs)
        x_B : numpy array
            time series of recieved signal for node B. (Sampled at self.Fs)
        num_reflections : int
            number of times signal is reflected. 1 indicates direct path. 10 indicates direct and 9
            reflections. Default - 10
        '''
        xA = np.zeros(self.t.shape)
        xB = np.zeros(self.t.shape)
        
        
        for _, source in sources.iterrows():
            coord = np.array([source.X, source.Y, source.Z])
            
            # set incoherent flag
            if source.label == 'fin_incoherent':
                incoherent = True
            # Generate source signal
            if source.label == 'gauss':
                signals = rng.randn(len(self.t)) # ~N(0,1)
            elif (source.label == 'fin_model') | (source.label == 'fin_incoherent'):
                boost = 1 # manually changeable term for mixing whale with other sources
                dt = 1
                f0 = 30
                f1 = 15
                T = 20
                
                t = np.arange(0,dt,1/200)
                win = signal.windows.gaussian(len(t), 40)
                chirp = signal.chirp(t,f0,dt,f1)*win
                chirp_padded = np.zeros(T*200)
                
                # place chirp randomly in T window
                start_idx = np.random.randint(0,len(chirp_padded)-len(chirp)-1)
                chirp_padded[start_idx:start_idx+len(chirp)] = chirp
                n_tile = int(self.time_length*200/len(chirp_padded))
                signals = np.tile(chirp_padded, n_tile)
            else:
                raise Exception('Iinvalid source label')

            # Loop through all reflections
            for k in range(num_reflections):
                if k % 2 == 0:
                    depth = k*self.depth + source.Z
                else:
                    depth = k*self.depth + (self.depth - source.Z)
                
                # get radius and time shift
                rA, rB = self.__get_radius([source.X, source.Y, depth])
                dt_A = rA/self.c
                dt_B = rB/self.c
 
                # create interpolation
                f = interpolate.interp1d(self.t, signals, kind='cubic', bounds_error=False)
            
                # interpolate time shift
                xA_single = f(self.t - dt_A)/(rA**2)
                xB_single = f(self.t - dt_B)/(rB**2)

                # remove spherical spreading if radius < 1 m
                if rA < 1:
                    xA_single = xA_single*(rA**2)
                if rB < 1:
                    xB_single = xB_single*(rB**2)

                # remove nan
                xA_single[np.isnan(xA_single)] = 0
                xB_single[np.isnan(xB_single)] = 0

                # handle incoherent sources
                if incoherent:
                    if source.hydrophone == 'A':
                        xB_single = np.zeros(len(self.t))
                    elif source.hydrophone == 'B':
                        xA_single = np.zeros(len(self.t))
                    else:
                        raise Exception('Invalid (or missing) hydrophone label')

                xA += xA_single
                xB += xB_single

        return xA, xB

    def get_signals(self):
        '''
        generates recieved signal at node a and node b given the environment
        Parameters
        ----------
        
        Returns
        -------
        xA : numpy array
            time series of signal recieved at node A
        xB : numpy array
            time series of signal recieved at node B
        '''
        sources = self.sources
        # remove whale sources if no_whale

        num_processes = mp.cpu_count()
        if len(sources) < num_processes:
            xA, xB = self.get_signals_1cpu(sources)
        else:
            # calculate the chuck size as an integer
            chunk_size = int(sources.shape[0]/(num_processes))
            # Divide dataframe up into num_processes chunks
            chunks = [sources.iloc[i:i + chunk_size,:] for i in range(0, sources.shape[0], chunk_size)]
            rngs = [np.random.RandomState(i) for i in range(num_processes)]
            temp = len(chunks)
            for k in range(num_processes, temp):
                chunks[num_processes-1] = chunks[num_processes-1].append(chunks[k])

            del chunks[num_processes:]

            # Original Method
            Pool = mp.Pool(processes = num_processes)
            results = Pool.starmap(self.get_signals_1cpu, zip(chunks, rngs)) # change back to chunks
            # Unpack result
            xA = np.zeros(self.t.shape)
            xB = np.zeros(self.t.shape)
            for k in range(num_processes-2):
                xA += results[k][0]
                xB += results[k][1]
        self.xA = xA
        self.xB = xB

        self.t_nccf = np.linspace(-self.time_length, self.time_length, len(xA)*2-1)
        return xA, xB

    def correlate(self, plot=False, ax=None):
        '''
        computes noise cross correlation function for generated signals xA and
            xB
        Parameters
        ----------
        plot : bool
            specifies whether to plot correlation
        ax : matplotlib.axis
            if specified, this axis is used
        Returns
        -------
        NCCF : numpy array
            noise cross correlation function
        '''
        xA = self.xA
        xB = self.xB
        NCCF = signal.fftconvolve(xA, np.flip(xB), mode='full')

        self.NCCF = NCCF
        if plot:
            if ax == None:
                fig, ax = plt.subplots(1,1,figsize=(7,5))
            ax.plot(self.t_nccf, NCCF)
            ax.set_xlim([-10,10])
            ax.set_xlabel('delay (s)')
            ax.grid()
        return NCCF


class source_distribution:
    def __init__(self, depth=1500):
        self.depth = 1500
    
    def single_point(self, x, y, z, label):
        '''
        single_point creates a single source at point (x,y,z) with label 'label'
        '''
        sources_dict = {'X':x, 'Y':y, 'Z':z, 'label':label}
        self.sources = pd.DataFrame(sources_dict, index=[0])

        return self.sources

    def surface_line(self, xmin=-10000, xmax=10000, npts = 1000, y=0, label='gauss'):
        '''
        single line in same dimension as nodes
        '''

        x = np.linspace(xmin, xmax, npts)
        y = np.ones(npts)*y
        z = np.ones(npts)*self.depth

        sources_dict = {'X':x, 'Y':y, 'Z':z, 'label':label}
        self.sources = pd.DataFrame(sources_dict)

        return self.sources

    def surface_line_incoherent(self, xmin=-10000, xmax=10000, npts = 1000, y=0, label='gauss', hydrophone='A'):
        '''
        single line in same dimension as nodes
        '''

        x = np.linspace(xmin, xmax, npts)
        y = np.ones(npts)*y
        z = np.ones(npts)*self.depth

        sources_dict = {'X':x, 'Y':y, 'Z':z, 'label':label, 'hydrophone':hydrophone}
        
        try:
            self.sources
            self.sources = pd.concat((self.sources, pd.DataFrame(sources_dict)), ignore_index=True)
        except:
            self.sources = pd.DataFrame(sources_dict)

        return self.sources