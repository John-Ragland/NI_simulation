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
        '''
        # Define Environment Variabels
        self.Fs = 200
        f0 = 50 #Hz
        self.w0 = 2*np.pi*f0
        self.sigma = 1/(5*f0)

        self.c = 1500 # m/s
        
        self.time_length = time_length
        self.t = np.arange(0,self.time_length, 1/self.Fs)
        self.nodeA = (-1500, 0)
        self.nodeB = (1500, 0)

        self.sources = sources

    def get_signals_1cpu(self, sources, rng=np.random.RandomState(0)):
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
        '''
        xA = np.zeros(self.t.shape)
        xB = np.zeros(self.t.shape)
        for _, source in sources.iterrows():
            coord = (source.X, source.Y)
            
            # get radius and time shift
            rA, rB = self.__get_radius(coord)
            dt_A = rA/self.c
            dt_B = rB/self.c
            if source.label == 'gauss':
                # Generate instance of gaussian noise N(0,1)
                noise = rng.randn(len(self.t))
                
                # create interpolation
                f = interpolate.interp1d(self.t, noise, kind='cubic', bounds_error=False)
                
                # interpolate time shift
                xA_single = f(self.t - dt_A)/(rA**2)
                xB_single = f(self.t - dt_B)/(rB**2)
           
            elif source.label == 'sin':
                boost = 1
                xA_single = boost*np.sin(2*np.pi*20*(self.t-dt_A))/(rA**2)
                xB_single = boost*np.sin(2*np.pi*20*(self.t-dt_B))/(rA**2)
            
            elif source.label == 'fin':
                boost = 120

                # read fin wav file and convert to Fs = 200Hz
                _, fin = wavfile.read('./atlfin_128_64_0-50-FinWhaleAtlantic-10x.wav')
                fin = fin/np.max(fin)
                fin200 = signal.decimate(fin, 4, zero_phase=True)

                # repeat in time to match time_length
                if len(self.t)/len(fin200) < 1:
                    fin_expanded = fin200[:len(self.t)]
                else:
                    fin_expanded = np.tile(fin200, (np.round(len(self.t)/len(fin200),)))[:len(self.t)]
                f = interpolate.interp1d(self.t, fin_expanded, kind='cubic', bounds_error=False)
                xA_single = boost*f(self.t - dt_A)/(rA**2)
                xB_single = boost*f(self.t - dt_B)/(rB**2)
            
            elif source.label == 'fin_model':
                boost = 3 # boosts signal by factor
                # Fin Whale Model Attributes
                dt = 0.3
                f0 = 25
                f1 = 15
                T = 20
                decay = -1
                
                # create time sequence
                t = np.arange(0,dt,1/200)
                win = np.exp(t*decay)
                chirp = signal.chirp(t,f0,dt,f1)*win

                chirp_padded = np.zeros(T*200)
                chirp_padded[:len(chirp)] = chirp
                n_tile = int(self.time_length*200/len(chirp_padded))
                chirp_extended = np.tile(chirp_padded, n_tile)
                f = interpolate.interp1d(self.t, chirp_extended, kind='cubic', bounds_error=False)
                xA_single = boost*f(self.t - dt_A)/(rA**2)
                xB_single = boost*f(self.t - dt_B)/(rB**2)

            else:
                raise Exception('Invalid source label')

            # remove spherical spreading if radius < 1 m
            if rA < 1:
                xA_single = xA_single*(rA**2)
            if rB < 1:
                xB_single = xB_single*(rB**2)

            # remove nan
            xA_single[np.isnan(xA_single)] = 0
            xB_single[np.isnan(xB_single)] = 0

            xA += xA_single
            xB += xB_single
            #print(f'{index/len(sources)*100:0.3}', end='\r')

            #print('.', end='')
        #print(mp.current_process().pid)
        return xA, xB

    def __get_radius(self, coord):
        '''
        gets radius to A and B for given coord

        Parameters
        ----------
        coord : tuple (length 2)
            (X,Y) coordinate of source to calculate radius to
        Returns
        -------
        rA : float
            radius from coord to A
        rB : float
            radius from coord to B
        '''
        rA = ((self.nodeA[0] - coord[0])**2 + (self.nodeA[1] - coord[1])**2)**0.5
        rB = ((self.nodeB[0] - coord[0])**2 + (self.nodeB[1] - coord[1])**2)**0.5
        return rA, rB

    def get_signals(self, no_whale = False, whale_sources=None):
        '''
        generates recieved signal at node a and node b given the environment
        Parameters
        ----------
        no_whale : bool
            specifies whether or not to add whale contributions
        
        Returns
        -------
        xA : numpy array
            time series of signal recieved at node A
        xB : numpy array
            time series of signal recieved at node B
        '''
        sources = self.sources
        # remove whale sources if no_whale
        if no_whale:
            sources = sources[sources.label == 'gauss']
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

            # TQDM
            '''
            with mp.Pool(num_processes) as p:
                result = list(tqdm.tqdm(p.imap(self.get_signals_1cpu, chunks), total=len(sources)))
            '''

            '''
            results = []
            with Pool(processes=2) as p:
                max_ = 30
                with tqdm.tqdm(total=max_) as pbar:
                    for i, result in enumerate(p.imap_unordered(self.get_signals_1cpu, chunks, chunksize=chunk_size)):
                        pbar.update()
                        results.append(result)
            '''

            # Original Method
            Pool = mp.Pool(processes = num_processes)
            counter = 0
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

    def add_incoherent(self, to_xA=True, to_xB=True, scale=1):
        
        if to_xA:
            self.xA += scale*np.random.normal(0, 1, len(self.xA))
        if to_xB:
            self.xB += scale*np.random.normal(0, 1, len(self.xB))

        return self.xA, self.xB

    def add_filtered_incoherent(self, b, a, to_xA=True, to_xB=True, scale=1):
        '''
        add filtered incoherent noise to xa, xb or both
        Parameters
        ----------
        b : np.array
            filter numerator coefficients
        a : np.array
            filter denominator coefficients
        '''
        
        if to_xA:
            noise = np.random.normal(0, scale, len(self.xA))
            noise = signal.lfilter(b,a,noise)
            self.xA += scale * noise
        if to_xB:
            noise = np.random.normal(0, scale, len(self.xB))
            noise = signal.lfilter(b,a,noise)
            self.xB += scale * noise

        return self.xA, self.xB

    def plot_env(self, whale_sources=None):
        try:
            (whale_sources == None)
            all_sources = self.sources.append(whale_sources)
        except:
            pass
        
        noise_sources_x = all_sources[all_sources.label == 'gauss'].X.to_numpy()
        noise_sources_y = all_sources[all_sources.label == 'gauss'].Y.to_numpy()
        
        sin_sources_x = all_sources[(all_sources.label == 'fin_model') | (all_sources.label == 'fin')].X.to_numpy()
        sin_sources_y = all_sources[(all_sources.label == 'fin_model') | (all_sources.label == 'fin')].Y.to_numpy()

        fig, ax = plt.subplots(1,1, figsize=(7,7))
        # Plot Gaussian Noise Sources
        ax.plot(noise_sources_x, noise_sources_y, '.')

        # Plot Sine Noise Sources
        ax.plot(sin_sources_x, sin_sources_y, '.', color = 'C1', markersize=15)

        # Plot hydrophones
        ax.plot(self.nodeA[0], self.nodeA[1], '.', color = 'r', markersize=20)
        ax.plot(self.nodeB[0], self.nodeB[1], '.', color = 'r', markersize=20)

        leg_elements = [
            Line2D(
                [0],[0], marker='o', color='w', label='Sources',
                markerfacecolor='C0', markersize=10),
            Line2D(
                [0],[0], marker='o', color='w', label='Hydrophone Nodes',
                markerfacecolor='r', markersize=10),
            Line2D(
                [0],[0], marker='o', color='w', label='Fin Whale Source',
                markerfacecolor='C1', markersize=10)
        ]
        
        ax.legend(handles=leg_elements, loc='lower right', fontsize=16)
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Distance (m)')
        plt.grid()
        return fig, ax

    def correlate(self, whale=False, just_whale=False, plot=False):
        '''
        computes noise cross correlation function for generated signals xA and
            xB
        Parameters
        ----------
        whale : bool
            specifies whether to correlate self.xA and self.sB or self.xA_whale and self.xB_whale
        Returns
        -------
        NCCF : numpy array
            noise cross correlation function
        '''
        if whale:
            xA = self.xA_whale
            xB = self.xB_whale
        else:
            xA = self.xA
            xB = self.xB
        NCCF = signal.fftconvolve(xA, np.flip(xB), mode='full')

        if just_whale:
            xA = self.xA_whale - self.xA
            xB = self.xB_whale - self.xB

        self.NCCF = NCCF
        if plot:
            fig = plt.figure(figsize=(7,5))
            plt.plot(self.t_nccf, NCCF)
            plt.xlim([-5,5])
            plt.xlabel('delay (s)')
            plt.grid()
            return NCCF, fig
        return NCCF

    def spectrogram(self):
        f, t, Sxx = signal.spectrogram(self.NCCF, fs=200, nperseg=32, noverlap=31, nfft=256)
        t = np.linspace(-self.time_length, self.time_length, len(t))
        spec = xr.DataArray(Sxx, dims=['frequency','time'], coords={'frequency':f, 'time':t})
        return spec

    def add_whale_signals(self, whale_sources):
        '''
        takes noise signals generated with get_signals(no_whale=True) and adds
            whale signals. This accomplished multiple whale experiements with
            all other variables remaining constant
        Parameters
        ----------
        whale_sources : pandas DataFrame
            a seperately created data frame for the whale distribution
        '''
        # check if object state is correct
        try:
            self.xA_whale
            raise Exception('Must get noise signals first without whale, run method get_signals(no_whale=True)')
        except:
            pass
        
        # add whale signals
        xA_justwhale, xB_justwhale = self.get_signals_1cpu(whale_sources)
        self.xA_whale = self.xA + xA_justwhale
        self.xB_whale = self.xB + xB_justwhale
        return xA_justwhale, xB_justwhale

class source_distribution2D:
    def __init__(self):
        pass

    def uniform_circular(self, radius, center, n_sources):
        '''
        uniform_circular creates a source distribution that is uniformly
            spaced along a circle with given radius and center and given
            number of sources
        
        Parameters
        ----------
        radius : float
            radius of circle sources are located on
        center : tuple (length 2)
            (x,y) cooridinate of circle center
        n_sources : float
            number of sources that will be used in simulation
        
        Returns
        -------
        sources : pandas DataFrame
            dataframe listing all sources and x, y cooridinates given in meters
        '''
        thetas = np.linspace(0, 2*np.pi, n_sources)

        x_coord = radius*np.cos(thetas) + center[0]
        y_coord = radius*np.sin(thetas) + center[1]
        
        labels = ['gauss']*len(x_coord)
        sources_dict = {'X':x_coord, 'Y':y_coord, 'label':labels}
        self.sources = pd.DataFrame(sources_dict)

        return self.sources

    def uniform(self, x_bound, y_bound, n_sources):
        '''
        creates uniform source distribution in space consisting of n_sources
            discrete sources

        Parameters
        ----------
        x_bound : float
            sources are uniformly distributed between +- x_bound
        y_bound : float
            sources are uniformly distributed between += y_bound
        n_sources : int
            number of sources included in simulation

        Returns
        -------
        sources : pandas.DataFrame
            list of source x and y cooridinates
        '''
        x = np.random.uniform(-x_bound, x_bound, n_sources)
        y = np.random.uniform(-y_bound, y_bound, n_sources)
        labels = ['gauss']*len(x)

        sources_dict = {'X':x, 'Y':y, 'label':labels}
        self.sources = pd.DataFrame(sources_dict)

        return self.sources
    
    def endfire_circle(self, deg_bound, radius, n_sources):
        thetas1 = np.linspace(-np.deg2rad(deg_bound), np.deg2rad(deg_bound), int(n_sources/2))
        thetas2 = np.linspace(np.pi-np.deg2rad(deg_bound), np.pi+np.deg2rad(deg_bound), int(n_sources/2))

        x_coord = radius*np.cos(thetas1)
        x_coord = np.hstack((x_coord, radius*np.cos(thetas2)))

        y_coord = radius*np.sin(thetas1)
        y_coord = np.hstack((y_coord, radius*np.sin(thetas2)))

        labels = ['gauss']*len(x_coord)

        sources_dict = {'X':x_coord, 'Y':y_coord, 'label':labels}
        self.sources = pd.DataFrame(sources_dict)

        return self.sources

    def distant_uniform(self, inner_radius, outer_radius, n_sources):
        
        x_coord = []
        y_coord = []

        while len(x_coord) < n_sources:
            x = np.random.uniform(-outer_radius, outer_radius, 1)
            y = np.random.uniform(-outer_radius, outer_radius, 1)

            if ((x**2 + y**2)**0.5 < inner_radius) | ((x**2 + y**2)**0.5 > outer_radius):
                pass
            else:
                x_coord.append(x)
                y_coord.append(y)
        
        labels = ['gauss']*len(x_coord)
        sources_dict = {'X':x_coord, 'Y':y_coord, 'label':labels}
        self.sources = pd.DataFrame(sources_dict)
        return self.sources

    def add_custom_sources(self, label, coord):
        '''
        adds sources with 20 Hz sine wave sources
        Parameters
        ----------
        label : str
            - 'sin'
            - 'fin'
            - 'model_fin'
        coord : tuple
            (r,θ) coordinate of source (in meters and degrees)
        Returns
        -------
        Updates Source Class
        '''
        # Check if noise sources exists
        try:
            self.sources
            first_source = False
        except AttributeError:
            first_source = True
        
        x = coord[0]*np.cos(np.deg2rad(coord[1]))
        y = coord[0]*np.sin(np.deg2rad(coord[1]))
        
        
        if first_source:
            sine_sources = {'X':[x], 'Y':[y], 'label':label}
            self.sources = pd.DataFrame(sine_sources)
        else:
            sine_sources = {'X':x, 'Y':y, 'label':label}
            self.sources = self.sources.append(sine_sources, ignore_index=True)

    def fin_whale_dist(self, inner_radius, outer_radius, deg_bound, n_sources, deg=180):

        r = outer_radius*np.sqrt(np.random.uniform((inner_radius/outer_radius)**2, 1, n_sources))
        theta = np.random.uniform(np.deg2rad(deg)-np.deg2rad(deg_bound), np.deg2rad(deg)+np.deg2rad(deg_bound), n_sources)
        x = r*np.cos(theta)
        y = r*np.sin(theta)       
        
        labels = ['fin_model']*len(x)
        sources_dict = {'X':x, 'Y':y, 'label':labels}
        try:
            self.sources = self.sources.append(pd.DataFrame(sources_dict))
        except:
            self.sources = pd.DataFrame(sources_dict)
        
        return self.sources