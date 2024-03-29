from typing import Counter
import numpy as np
from numpy.core.numeric import outer
import scipy
import pandas as pd
from scipy import interpolate
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm

from scipy.io import wavfile
import scipy.io
from scipy import signal

from matplotlib.lines import Line2D
from matplotlib import pyplot as plt

class environment:
    def __init__(self, sources, time_length=60, frequencies = None, Fs = 200):
        '''
        initialize environment variable
        Parameters
        ----------
        sources : pandas.DataFrame
            dataframe of source x and y cooridinates
        '''
        # Define Environment Variabels
        self.Fs = Fs
        f0 = 50 #Hz
        self.w0 = 2*np.pi*f0
        self.sigma = 1/(5*f0)

        self.c = 1500 # m/s
        
        self.time_length = time_length
        self.t = np.arange(0,self.time_length, 1/self.Fs)
        self.nodeA = (-3186/2, 0)
        self.nodeB = (3186/2, 0)

        self.sources = sources
        
        self.frequencies = frequencies
        self.tau = np.linspace(-self.time_length + 1/self.Fs, self.time_length - 1/self.Fs, 2*self.time_length*self.Fs -1)

    def __get_time_signal(self, r):
        '''
        get_time_signal constructs time signal for a single hydrophone and
            single source. These signals can then be superimposed for multiple
            sources and one hydrophone
        Parameters
        ----------
        r : float
            distance between given source and hydrophone
        
        Returns
        -------
        x : numpy array
            sampled signal for single source reciever pair
        '''
        x = 1/(r**2)*np.exp(-((self.t-r/self.c)**2)/(2*self.sigma)**2)*np.exp(1j*self.w0*(self.t-r/self.c))
        return x

    def get_signals_1cpu(self, source):
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
        
        # set random seed to be source number
        # so that different cores don't generate same numbers
        np.random.seed(int(np.array(source.index)))
        
        coord = (source.X.values[0], source.Y.values[0])
            
        # get radius and time shift
        rA, rB = self.__get_radius(coord)
        dt_A = rA/self.c
        dt_B = rB/self.c
            
        if source.label.values[0] == 'gauss':
            # Generate instance of gaussian noise N(0,1)
            noise = np.random.normal(0,1,len(self.t))

            # create interpolation
            f = interpolate.interp1d(self.t, noise, kind='cubic', bounds_error=False)

            # interpolate time shift
            xA_single = f(self.t - dt_A)/(rA)
            xB_single = f(self.t - dt_B)/(rB)

        elif source.label.values[0] == 'sin':
            boost = 1
            xA_single = boost*np.sin(2*np.pi*20*(self.t-dt_A))/(rA**2)
            xB_single = boost*np.sin(2*np.pi*20*(self.t-dt_B))/(rA**2)

        elif source.label.values[0] == 'fin':
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

        elif source.label.values[0] == 'fin_model':
            boost = 1.5 # boosts signal by factor

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

        elif source.label.values[0] == 'harmonic':
            
            freqs = self.frequencies
            xA_single = np.zeros(self.t.shape)
            xB_single = np.zeros(self.t.shape)
            
            freq_range = np.max(freqs) - np.min(freqs)
            for freq in list(freqs):
                # x is sinusoid with time delay equal to dt and a phase
                # which is a function of frequency and frequency range
                # weird phase, but makes computations faster
                xA_single = xA_single + np.sin((self.t - dt_A)*2*np.pi*freq + (freq - np.min(freqs))*2*np.pi/freq_range)
                xB_single = xB_single + np.sin((self.t - dt_B) *2*np.pi*freq + (freq - np.min(freqs))*2*np.pi/freq_range)

        else:
            raise Exception('Invalid source label')

        # remove spherical spreading if radius < 1 m
        if rA < 1:
            xA_single = xA_single*(rA)
        if rB < 1:
            xB_single = xB_single*(rB)

        # remove nan
        xA_single[np.isnan(xA_single)] = 0
        xB_single[np.isnan(xB_single)] = 0

        return xA_single, xB_single

    def get_correlations_1cpu(self, source):
        '''
        get_correlations_1cpu calculated the time domain recieved signal at node A and B
            for a given source distribution and then correlate them
        
        Parameters
        ----------
        sources : pandas.DataFrame
            data frame containing the x and y information for every source
        
        Returns
        -------
        R : np.array
            cross correlation between signals recieved at A and B
        '''
        
        # set random seed to be source number
        # so that different cores don't generate same numbers
        np.random.seed(int(np.array(source.index)))
        
        coord = (source.X.values[0], source.Y.values[0])
            
        # get radius and time shift
        rA, rB = self.__get_radius(coord)
        dt_A = rA/self.c
        dt_B = rB/self.c
            
        if source.label.values[0] == 'gauss':
            # Generate instance of gaussian noise N(0,1)
            noise = np.random.normal(0,1,len(self.t))
            
            # filter noise to 5 hz
            b, a = signal.butter(4,10/100, btype='low')
            noise = signal.lfilter(b,a,noise)
            
            # create interpolation
            f = interpolate.interp1d(self.t, noise, kind='cubic', bounds_error=False)

            # interpolate time shift
            xA_single = f(self.t - dt_A)/(rA)
            xB_single = f(self.t - dt_B)/(rB)

        elif source.label.values[0] == 'sin':
            boost = 1
            xA_single = boost*np.sin(2*np.pi*20*(self.t-dt_A))/(rA**2)
            xB_single = boost*np.sin(2*np.pi*20*(self.t-dt_B))/(rA**2)

        elif source.label.values[0] == 'fin':
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

        elif source.label.values[0] == 'fin_model':
            boost = 1.5 # boosts signal by factor

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

        elif source.label.values[0] == 'harmonic':
            
            freqs = self.frequencies
            xA_single = np.zeros(self.t.shape)
            xB_single = np.zeros(self.t.shape)
            
            freq_range = np.max(freqs) - np.min(freqs)
            for freq in list(freqs):
                # x is sinusoid with time delay equal to dt and a phase
                # which is a function of frequency and frequency range
                # weird phase, but makes computations faster
                xA_single = xA_single + np.sin((self.t - dt_A)*2*np.pi*freq + (freq - np.min(freqs))*2*np.pi/freq_range)
                xB_single = xB_single + np.sin((self.t - dt_B) *2*np.pi*freq + (freq - np.min(freqs))*2*np.pi/freq_range)

        else:
            raise Exception('Invalid source label')

        # remove spherical spreading if radius < 1 m
        if rA < 1:
            xA_single = xA_single*(rA)
        if rB < 1:
            xB_single = xB_single*(rB)

        # remove nan
        xA_single[np.isnan(xA_single)] = 0
        xB_single[np.isnan(xB_single)] = 0

        R = signal.fftconvolve(xA_single, np.flip(xB_single), mode='full')
        return R

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
    
    def get_correlations(self, correlation_type='single', verbose=True, chunksize=1):
        '''
        get_correlation - version of get_signals, but the correlations are
            taken before added seperate signal sources up.
            
            This forces the correlation between sources to be zero.
            
        Parameters
        ----------
        correlation_type : string
            decides whether to return sum of all cross correlations ('single')
            or every seperate cross_correlation ('all') default is single
        Returns
        -------
        R : np.array
            NCCF
        t : np.array
            delay time cooridinate for NCCF
        '''
        # check valid return type before simulation:
        if (correlation_type != ('single')) & (correlation_type != ('all')):
            raise Exception('invalid return type')
            
        sources = self.sources
        
        # set num_processes to be number of cores, or size of array
            # whichever is smaller
        
        if mp.cpu_count() < len(sources):
            num_processes = mp.cpu_count()
        else:
            num_processes = len(sources)
        
        # Divide dataframe up into list of rows
        chunks = [sources.iloc[i:i + 1,:] for i in range(0, sources.shape[0], 1)]
        # Original Method
        # print(num_processes)
        
        Pool = mp.Pool(processes = num_processes)
        result = list(tqdm(Pool.imap(self.get_correlations_1cpu, chunks, chunksize), total=len(self.sources), disable=(not verbose)))
        #result = list(Pool.imap(self.get_correlations_1cpu, chunks, chunksize))
    
        Pool.close()
        
        if correlation_type == 'all':
            R = result
        elif correlation_type == 'single':
            # Unpack result
            for k in range(len(result)):
                if k == 0:
                    R = result[k]
                else:
                    R += result[k]
        
        
        self.R = R
        return self.tau, self.R
    
    def get_signals(self):
        '''
        
        '''
        sources = self.sources
        
        # set num_processes to be number of cores, or size of array
            # whichever is smaller
        
        if mp.cpu_count() < len(sources):
            num_processes = mp.cpu_count()
        else:
            num_processes = len(sources)
        
        # Divide dataframe up into list of rows
        chunks = [sources.iloc[i:i + 1,:] for i in range(0, sources.shape[0], 1)]

        # Original Method
        Pool = mp.Pool(processes = num_processes)
        
        result = list(tqdm(Pool.imap(self.get_signals_1cpu, chunks), total=len(self.sources)))
        
        #result = Pool.map(self.get_signals_1cpu, chunks)
        Pool.close()

        # Unpack result
        xA = np.zeros(self.t.shape)
        xB = np.zeros(self.t.shape)
        for k in range(len(sources)):
            xA += result[k][0]
            xB += result[k][1]
       
        return xA, xB

    def get_signal_mp(self, x, y):
        '''
        get_signal_mp executes contents of what was a for loop using
            multiprocessing
        Parameters
        ----------
        source : pandas.DataFrame single row
            contains (x,y) cooridinates of source in meters
    
        Returns
        -------

        '''
        coord = (x, y)
        rA, rB = self.__get_radius(coord)
        xA_single, xB_single = self.__get_time_signals_guassian(rA, rB)

        return[xA_single, xB_single]

    def plot_env(self):
        noise_sources_x = self.sources[self.sources.label == 'gauss'].X.to_numpy()
        noise_sources_y = self.sources[self.sources.label == 'gauss'].Y.to_numpy()
        
        sin_sources_x = self.sources[(self.sources.label == 'fin_model') | (self.sources.label == 'fin')].X.to_numpy()
        sin_sources_y = self.sources[(self.sources.label == 'fin_model') | (self.sources.label == 'fin')].Y.to_numpy()

        fig, ax = plt.subplots(1,1, figsize=(7,7))
        # Plot Gaussian Noise Sources
        ax.plot(noise_sources_x, noise_sources_y, 'k.', markersize=1)

        # Plot Sine Noise Sources
        ax.plot(sin_sources_x, sin_sources_y, 'k.', markersize=7)

        # Plot hydrophones
        ax.plot(self.nodeA[0], self.nodeA[1], 'o', color = 'r', markersize=5)
        ax.plot(self.nodeB[0], self.nodeB[1], 'o', color = 'r', markersize=5)

        leg_elements = [
            Line2D(
                [0],[0], marker='o', color='w', label='Sources',
                markerfacecolor='k', markersize=8),
            Line2D(
                [0],[0], marker='o', color='w', label='Hydrophone Nodes',
                markerfacecolor='r', markersize=8),
            #Line2D(
            #    [0],[0], marker='o', color='w', label='Fin Whale Source',
            #    markerfacecolor='C1', markersize=10)
        ]
        
        ax.legend(handles=leg_elements, loc='lower right', fontsize=16)
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Distance (m)')
        plt.grid()
        return fig, ax

    def correlate(self):
        '''
        computes noise cross correlation function for generated signals xA and
            xB
        
        Returns
        -------
        NCCF : numpy array
            noise cross correlation function
        '''
    
        xA = self.xA
        xB = self.xB
        NCCF = signal.fftconvolve(xA, np.flip(xB), mode='full')

        return NCCF

    def directly_sim_NCCF(self, fc, correlation_type='single'):
        '''
        directly_sim_NCCF - simulate the theoretical NCCF using the peak location caused by the
            difference in ranges between the source and H1 and H2. A Gaussian pulse is used
            TODO : code to solve for pulse width given cutoff frequency is brokem
        
        Parameters
        ----------
        fc : float
            cutoff frequency (in Hertz) of guassian pulse
            
        correlation_type : str
            'single' or 'all'
            determines wheter to return signle correlation or all correaltions
        
        Returns
        -------
        R : np.array
            numpy array of shape (len(sources), len(tau)) or
            of shape (len(tau)) depending on correlation_type        
        '''
        
        rs = np.vstack((self.sources.X, self.sources.Y))
        r1 = np.matlib.repmat(np.expand_dims(np.array(self.nodeA), 1), 1, len(self.sources))
        r2 = np.matlib.repmat(np.expand_dims(np.array(self.nodeB), 1), 1, len(self.sources))
        R1 = np.linalg.norm(r1 - rs, axis=0)
        R2 = np.linalg.norm(r2 - rs, axis=0)
        Rd = R1 - R2
        a = -np.pi**2*fc/np.log(1/np.sqrt(2))

        if correlation_type == 'single':
            x = np.zeros(self.tau.shape)

            for k in tqdm(range(len(self.sources))):
                x1 = np.sqrt(a/np.pi)*np.exp(-a*(self.tau-Rd[k]/self.c)**2)
                x = x + x1
        elif correlation_type == 'all':
            x = np.zeros((len(self.sources), len(self.tau)))
            
            for k in tqdm(range(len(self.sources))):
                x1 = np.sqrt(a/np.pi)*np.exp(-a*(self.tau-Rd[k]/self.c)**2)
                x[k,:] = x1
        else:
            raise Exception("invalid correlation_type string should ['single', 'all']")
        
        return x
            
class source_distribution2D:
    def __init__(self):
        pass

    def uniform_circular(self, radius, center, n_sources, label='harmonic'):
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
        label : str
            determines how the acoustic field is calculated
            
        Returns
        -------
        sources : pandas DataFrame
            dataframe listing all sources, x, y cooridinates given in meters
            and the label (which determines how to calculate the acoustic field)
        '''
        thetas = np.linspace(0, 2*np.pi, n_sources)

        x_coord = radius*np.cos(thetas) + center[0]
        y_coord = radius*np.sin(thetas) + center[1]
        
        labels = ['gauss']*len(x_coord)
        sources_dict = {'X':x_coord, 'Y':y_coord, 'label':labels}
        self.sources = pd.DataFrame(sources_dict)

        return self.sources

    def uniform(self, x_bound, y_bound, n_sources, label='harmonic'):
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
        label : str
            determines how the acoustic field is calculated
            
        Returns
        -------
        sources : pandas DataFrame
            dataframe listing all sources, x, y cooridinates given in meters
            and the label (which determines how to calculate the acoustic field)
        '''
        x = np.random.uniform(-x_bound, x_bound, n_sources)
        y = np.random.uniform(-y_bound, y_bound, n_sources)
        labels = ['gauss']*len(x)

        sources_dict = {'X':x, 'Y':y, 'label':labels}
        self.sources = pd.DataFrame(sources_dict)

        return self.sources
    
    def endfire_circle(self, deg_bound, radius, n_sources, label='harmonic'):
        '''
        endfire_circle creates a source distribution that is in the endfire
            direction a circle with given radius and given
            number of sources
        
        Parameters
        ----------
        deg_bound : float
            deg bound of circle
        radius : float
            radius of circle
        n_sources : float
            number of sources that will be used in simulation
        label : str
            determines how the acoustic field is calculated
        
        Returns
        -------
        sources : pandas DataFrame
            dataframe listing all sources, x, y cooridinates given in meters
            and the label (which determines how to calculate the acoustic field)
        '''
        
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

    def distant_uniform(self, inner_radius, outer_radius, n_sources, label='harmonic'):
        '''
        distant creates a source distribution that is uniformly
            spaced along a circle with given radius and center and given
            number of sources
        
        Parameters
        ----------
        inner_radius : float
            inner radius of distribution
        outer_radius : float
            outer radius of distribution
        n_sources : float
            number of sources that will be used in simulation
        label : str
            determines how the acoustic field is calculated
            
        Returns
        -------
        sources : pandas DataFrame
            dataframe listing all sources, x, y cooridinates given in meters
            and the label (which determines how to calculate the acoustic field)
        '''
        x_coord = []
        y_coord = []

        while len(x_coord) < n_sources:
            x = np.random.uniform(-outer_radius, outer_radius, 1)[0]
            y = np.random.uniform(-outer_radius, outer_radius, 1)[0]

            if ((x**2 + y**2)**0.5 < inner_radius) | ((x**2 + y**2)**0.5 > outer_radius):
                pass
            else:
                x_coord.append(x)
                y_coord.append(y)
        
        labels = [label]*len(x_coord)
        sources_dict = {'X':x_coord, 'Y':y_coord, 'label':labels}

        sources = pd.DataFrame(sources_dict)
        return sources
    
    def add_custom_sources(self, x,y,label):
        '''
        adds sources with 20 Hz sine wave sources
        Parameters
        ----------
        x : array like
            x position of sources in meters
        y : array like 
            y position of sources in meters
        label : str
            - 'sin'
            - 'fin'
            - 'model_fin'
            - 'gauss'
        
        Returns
        -------
        Updates Source Class
        '''

        custom_sources = {'X':[x], 'Y':[y], 'label':label}

        # Check if noise sources exists
        try:
            self.sources
            self.sources = self.sources.append(custom_sources, ignore_index=True)
        
        except AttributeError:
            self.sources = pd.DataFrame(data = custom_sources)
            
        return self.sources
        
    def fin_whale_dist(self, inner_radius, outer_radius, deg_bound, n_sources, deg=180):

        r = outer_radius*np.sqrt(np.random.uniform((inner_radius/outer_radius)**2, 1, n_sources))
        theta = np.random.uniform(np.deg2rad(deg)-np.deg2rad(deg_bound), np.deg2rad(deg)+np.deg2rad(deg_bound), n_sources)
        x = r*np.cos(theta)
        y = r*np.sin(theta)       
        
        labels = ['fin_model']*len(x)
        sources_dict = {'X':x, 'Y':y, 'label':labels}
        self.sources = self.sources.append(pd.DataFrame(sources_dict))
        
        return self.sources
      
    def spiral(self, theta_range, range_range, npts, label='gauss'):
        '''
        creates spiral source distribution with mesh grid of 
            thetas and ranges
            
            cartesian conversion is done with conventional
                polar coordinate definition (theta is from x)
                
        Parameters
        ----------
        npts : int
            total number of sources
            sqrt(npts) must be integer. If not, it's forced
        theta_range : tuple
            min and max of theta
        range_range : tuple
            min and max of range
        '''
        
        thetas = list(np.linspace(theta_range[0], theta_range[1], int(np.sqrt(npts))))
        ranges = list(np.linspace(range_range[0], range_range[1], int(np.sqrt(npts))))
        
        x_coord = []
        y_coord = []
        
        for theta in thetas:
            for range_single in ranges:
                x_coord.append(range_single*np.cos(theta))
                y_coord.append(range_single*np.sin(theta))
                
        labels = [label]*len(x_coord)
        sources_dict = {'X':x_coord, 'Y':y_coord, 'label':labels}

        sources = pd.DataFrame(sources_dict)
        return sources

def calc_directivity(x, y, R, radius):
    '''
    calc_directivity - calculates the directivity pattern of the NCCF
        for a given noise source distribution.

    Parameters
    ----------
    x : array like
        x dimension location of sources
    y : array like
        y dimension location of sources
    R : array like
        correlations due to point sources at (x,y). Should have shape
        (len(x), len(tau)) where tau is delay time of correlations
    radius : float
        radius (in meters) to calculate the directivity
    
    Returns
    -------
    theta : np.array
        angle bins for directivity
    directivity : np.array
        directivity pattern for given source distribution
    '''    
    
    print('calculating correlation coefficents...')
    rhos = calc_corr_coef(R)
    print('calculating surface interpolation...')
    f = scipy.interpolate.interp2d(x, y, rhos, kind='cubic')
    
    theta = np.linspace(0,2*np.pi, 1000)
    x_new = radius*np.cos(theta)
    y_new = radius*np.sin(theta)
    
    directivity = f(x_new, y_new)
    
    return theta, directivity
    
def calc_corr_coef(R, R_ens, plot=False, sources=None):
    '''
    calc_corr_coef - calculates the correlation coefficent for 
        a single source NCCF and the ensemble NCCF
    
    Parameters
    ----------
    R : np.array
        array of shape (len(sources), len(tau)). created by 
        environment.directly_sim_NCCF(correlation_type='all')
    R_ens : np.array
        array of shape(len(tau)). this can either be the ensemble
        cross correlation from all sound sources, or it can be
        the theoretical TDGF between the two points
    plot : bool
        determines whether to plot correlation coefficient
        
    Returns
    -------
    R_corr : np.array
        correlation of single NCCF to ensemble NCCF
    '''
    rhos = []
    for k in tqdm(range(R.shape[0])):
        rho,_ = scipy.stats.pearsonr(R_ens, R[k,:])
        rhos.append(rho)
    rhos = np.array(rhos)
    
    if plot:
        plt.figure(figsize=(11,9))
        plt.tricontourf(sources.X, sources.Y, rhos, 1000, cmap='plasma')
        plt.colorbar()
    return rhos
