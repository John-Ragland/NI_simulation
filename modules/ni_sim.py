import numpy as np
import scipy
import pandas as pd
from scipy import interpolate
import multiprocessing as mp
from multiprocessing import Pool
import tqdm

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

    def get_signals_1cpu(self, sources):
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
        for index, source in self.sources.iterrows():
            coord = (source.X, source.Y)
            rA, rB = self.__get_radius(coord)
            
            # Generate instance of gaussian noise N(0,1)
            noise = np.random.normal(0,1,len(self.t))

            # create interpolation
            f = interpolate.interp1d(self.t, noise, kind='cubic', bounds_error=False)

            dt_A = rA/self.c
            dt_B = rB/self.c

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

    def get_signals(self):
        sources = self.sources
        num_processes = mp.cpu_count()

        # calculate the chuck size as an integer
        chunk_size = int(sources.shape[0]/(num_processes-1))

        # Divide dataframe up into num_processes chunks
        #chunks = [sources.ix[sources.index[i:i + chunk_size]] for i in range(0, sources.shape[0],chunk_size)]
        chunks = [sources.iloc[i:i + chunk_size,:] for i in range(0, sources.shape[0], chunk_size)]

        # TQDM
        '''
        with mp.Pool(num_processes) as p:
            result = list(tqdm.tqdm(p.imap(self.get_signals_1cpu, chunks), total=len(sources)))
        '''

        # Original Method
        self.count = 0
        Pool = mp.Pool(processes = num_processes)
        result = Pool.map(self.get_signals_1cpu, chunks)
        Pool.close()

        # Unpack result
        xA = np.zeros(self.t.shape)
        xB = np.zeros(self.t.shape)
        for k in range(num_processes):
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
        sources_x = self.sources.X.to_numpy()
        sources_y = self.sources.Y.to_numpy()

        fig, ax = plt.subplots(1,1, figsize=(7,7))
        ax.plot(sources_x, sources_y, '.')

        ax.plot(self.nodeA[0], self.nodeA[1], '.', color = 'r', markersize=20)
        ax.plot(self.nodeB[0], self.nodeB[1], '.', color = 'r', markersize=20)

        leg_elements = [
            Line2D([0],[0], marker='o', color='w', label='Sources',markerfacecolor='C0', markersize=10),
            Line2D([0],[0], marker='o', color='w', label='Hydrophone Nodes', markerfacecolor='r', markersize=10)]
        ax.legend(handles=leg_elements, loc='upper right', fontsize=16)
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Distance (m)')
        return fig, ax

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

        sources_dict = {'X':x_coord, 'Y':y_coord}
        sources = pd.DataFrame(sources_dict)

        return sources

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

        sources_dict = {'X':x, 'Y':y}
        sources = pd.DataFrame(sources_dict)

        return sources
    
    def endfire_circle(self, deg_bound, radius, n_sources):
        thetas1 = np.linspace(-np.deg2rad(deg_bound), np.deg2rad(deg_bound), int(n_sources/2))
        thetas2 = np.linspace(np.pi-np.deg2rad(deg_bound), np.pi+np.deg2rad(deg_bound), int(n_sources/2))

        x_coord = radius*np.cos(thetas1)
        x_coord = np.hstack((x_coord, radius*np.cos(thetas2)))

        y_coord = radius*np.sin(thetas1)
        y_coord = np.hstack((y_coord, radius*np.sin(thetas2)))

        sources_dict = {'X':x_coord, 'Y':y_coord}
        sources = pd.DataFrame(sources_dict)

        return sources

    def distant_uniform(self, inner_radius, x_bound, y_bound, n_sources):
        
        x_coord = []
        y_coord = []

        while len(x_coord) < n_sources:
            x = np.random.uniform(-x_bound, x_bound, 1)
            y = np.random.uniform(-y_bound, y_bound, 1)

            if (x**2 + y**2)**0.5 < inner_radius:
                pass
            else:
                x_coord.append(x)
                y_coord.append(y)
        
        sources_dict = {'X':x_coord, 'Y':y_coord}
        sources = pd.DataFrame(sources_dict)
        return sources