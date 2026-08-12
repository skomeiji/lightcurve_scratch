"""
Module for computing variability timescales.

@juliaroquette:

__Last Modified__: 28 July 2025
- Removed old StructureFunction class

"""

import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from iminuit import Minuit
import warnings
import matplotlib.pyplot as plt
from variability.lightcurve import LightCurve
from variability.structure_function import StructureFunction


class TimeScale_refactored:
    def __init__(self, lc, **kwargs):
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        self.lc = lc
        # initialize attributes
        # minimum frequency resolved by the light-curve while guaranteeing at least two full cycle are covered
        self.min_freq =  kwargs.get('min_freq', 2. / (max(self.lc.time) - min(self.lc.time)))
        # time differences between consecutive epochs
        self.dt = np.diff(np.sort(self.lc.time))
        # some statistics on the time differences
        self.median_dt = np.median(self.dt)
        self.mean_dt = np.mean(self.dt)
        self.geom_mean_dt = np.exp(np.mean(np.log(self.dt)))
        self.min_dt = np.min(self.dt)
        self.max_dt = np.max(self.dt)
        # average time difference between the 5 smallest time differences
        self.max_freq = kwargs.get('max_freq', 1./2./np.sort(self.dt)[:5].mean())

    def get_GLS_periodogram(self, method='slow', samples_per_peak=5, normalization='standard'):
        ls = LombScargle(self.time, self.mag, self.err, method=method, normalization=normalization)
        frequency, power = ls.autopower(samples_per_peak=samples_per_peak,
                                            minimum_frequency=self.min_freq,
                                            maximum_frequency=self.max_freq, 
                                            method=method) 
        self.freq = frequency
        self.power = power
    
    def get_n_highest_peaks(self, N=5):
        self.get_GLS_periodogram()
        # Find indices of peaks in the power array
        freq_res = self.freq[1] - self.freq[0]
        df = 1 / (max(self.lc.time) - min(self.lc.time))
        peaks, _ = find_peaks(self.power, distance=int(5*freq_res/df))
        # Get the powers at those indices and sort to get the top N
        top_n_indices = peaks[np.argsort(self.power[peaks])[-N:]][::-1]
        # Print the corresponding frequencies and powers
        for i, idx in enumerate(top_n_indices):
            vars(self)[f'gls_freq_{i + 1}'] = self.freq[idx]
            vars(self)[f'gls_power_{i + 1}'] = self.power[idx]



class TimeScale:
    def __init__(self, lc, **kwargs):
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        self.lc = lc
        # if 'lc' in kwargs:
        #     # can take a LightCurve object as input
        #     super().__init__(kwargs['lc'].time, kwargs['lc'].mag, kwargs['lc'].err)
        #     self.lc = kwargs['lc']
        # elif all(key in kwargs for key in ['time', 'mag', 'err']):
        #     # otherwise, can take time, mag and err arrays as input and define a LightCurve object
        #     super().__init__(kwargs['time'], kwargs['mag'], kwargs['err'], kwargs.get('mask', None))
        #     self.lc = LightCurve(self.time, self.mag, self.err)
        # else:
        #     raise ValueError("Either a LightCurve object or time, mag and err arrays must be provided")  
        # initialize attributes
        self.ts = np.nan
        self.method = None
        self.fap = np.nan
        self.power = np.nan
        self.LSP_ts = np.nan
        self.C0 = np.nan
        self.C1 = np.nan
        self.SF_ts = np.nan
        # deal with which timescale method to use
        # if method is SF or LSP, it will only return the timescale in the given method. Otherwise, it will attempt to get a timescale using LSP first, and then SF if no periodic timescale was found.
        if 'method' not in kwargs:
            warnings.warn("No 'method' specified. Using default 'auto' method.", UserWarning)
            kwargs['method'] = 'auto'
        elif kwargs['method'] not in ['LSP', 'SF', 'auto']:
            raise ValueError("Method must be 'LSP', 'SF' or 'auto'")
        method = kwargs['method'] 
        #
        fap_prob = kwargs.get('fap_prob', 0.01)
        definition = kwargs.get('definition', 'auto')
        osf, min_freq, max_freq = pre_defined_parameters(self.lc.time, definition=definition)
        # if methods set to lomb scargle or to auto, first try to get 
        # timescales using the Lomb-Scargle periodogram
        if method in ['LSP', 'auto']:
            best_freq, best_power,\
            FAP_highest_peak = self.get_LSP_period(fmin=min_freq,
                                                   fmax=max_freq, 
                                                   osf=osf, 
                                                   periodogram=kwargs.get('periodogram', False), 
                                                   definition=definition)
            self.LSP_ts = 1 / best_freq
            self.fap = FAP_highest_peak
            self.power = best_power
            
            if (FAP_highest_peak < fap_prob) or (method =="LSP"):
                self.ts = 1.*self.LSP_ts 
                self.method = 'LSP'
        # if method is set to SF, or if no timescale was obtained from LS
        # then proceed to get timescale from SF
        if method == 'SF' or (method == 'auto' and (self.method is None)):
            try:
                # self.SF_ts, self.C0, self.C1 = self.get_structure_function_timescale()
                # self.method = 'SF'
                # self.ts = 1.*self.SF_ts
                print('Integrate new StructureFunction class')
                pass
            except Exception as e:
                warnings.warn(f"Structure Function failed: {e}")
                
    def get_LSP_period(self, 
                          fmin, 
                          fmax,
                          osf,
                          periodogram,
                          definition='auto'):
        """
        Simple Period estimation code using Lomb-Scargle. 
        This adopts an heuristic approach for the frequency grid where,
        given the max/min values in the frequency, the grid is oversampled 
        by a default value of 5.
        

        Args:
            astropy.timeseries.LombScargle arguments:
            osf (int, optional): samples_per_peak Defaults to 5.
            fmin (int, optional): minimum frequency for the periodogram
            fmax (int, optional): maximum frequency for the periodogram
                NOTE: default min and max value consider:
                        - fmax is set by a 0.5 days period, which is about 
                        the break-up speed for very young stars. 
                        - fmin is arbitrary set to 250 days.
            periodogram (bool, optional): if True, returns the periodogram, 
                                          otherwise returns the period.

        Returns:
        if periodogram is True:
            frequency float: frequency of the highest peak
            power float: power of the highest peak
            FAP_level float: False alarm probability level for 1%, 10% and 40%
        else:
            frequency of the highest peak: float
            power of the highest peak: float
            FAP_highest_peak: 0-1. float: False Alarm Probability for the highest peak
        """
        # define the base for the Lomb-Scargle
        ls = LombScargle(self.lc.time, self.lc.mag, self.lc.err)
        if definition == 'Chloe':
            frequency = np.arange(fmin, fmax, step=0.0002)
            power = ls.power(frequency, method='slow')
        else:
            frequency, power = ls.autopower(samples_per_peak=osf,
                                            minimum_frequency=fmin,
                                            maximum_frequency=fmax, 
                                            method='slow') 
            # note here that method="slow" gives the astropy equivalente of GLS

        self.FAP_probs = ls.false_alarm_probability(power,method='baluev', 
                                            minimum_frequency=fmin, 
                                            maximum_frequency=fmax, 
                                            samples_per_peak=osf)
        
        if bool(periodogram):
            return frequency, power, self.FAP_probs
        else:
            freq_highest_peak = frequency[np.argmax(power)]
            power_highest_peak = power.max()
            # print('Highest peak:', pow)
            # print('Frequency of highest peak:', freq_highest_peak)
            FAP_highest_peak = ls.false_alarm_probability(power_highest_peak,method='baluev', 
                                         minimum_frequency=fmin, 
                                         maximum_frequency=fmax, 
                                         samples_per_peak=osf)
            # print(FAP_highest_peak)
            return freq_highest_peak, power_highest_peak, FAP_highest_peak

    def get_structure_function_timescale(self):
        """
        Uses Chloé's implementation of the structure function to get a timescale
        """
        sf = StructureFunction(lc=self.lc)
        # estimate SF values
        sf.get_sf()
        # Needs to pass desired parameters to the binning method
        sf.bin_sf(sf_err=None,
                       log=True,
                       hybrid=False,
                       bin_min_size=5, 
                       max_bin_exp_factor=3.0, 
                       step_size=0.2, 
                       resolution=0.02)
        # fit the structure function - needs to pass fitting parameters
        sf.fit_sf(yerr=None, last_params=[1, 0.01, 0.1], # last fit parameters
                    limits = [(0.07, 1800), (1e-6, 5), (1e-5, 100)], #
                    log=True, cost_flavour='L2_error', reduced_chi2=True, input_cost=None)
        (ts, ts_error), C0, C1, cost = sf.get_timescale()
        # print('SF timescale:', sf.ts)
        return ts, ts_error
    

    
def pre_defined_parameters(time, definition='Gaia'):
    """
    Returns pre-defined parameters for the TimeScale class.
    
    What is defined:
    - ofs (samples per peak:)
        This factor is called samples_per_peak in Astropy's LombScargle
        and it defaults to 5. 
        Once fmin and fmax are defined, this is used to set the step in frequency for the .autopower method. 
        In Gaia, this is called `stepFrequencyScaleFactor` and had 
        recommended values of 5 in DR3 and 10 in DR4.
    - min_freq (minimum frequency):
        This is the minimum frequency for the periodogram.
    - max_freq (maximum frequency):
        This is the maximum frequency for the periodogram.
    """
    if definition == 'Gaia':
        osf = 10
        min_freq = 0.001
        max_freq = 2.8
    elif definition == 'Chloe':
        # Chloe defined that dynamically where she actually fiexes the df
        osf = 5
        min_freq = 1./ ((max(time) - min(time)))
        max_freq = 1 / (np.median(np.diff(time)) * 2)
    elif definition == 'auto':
        osf = 5
        # Guarantees at least one full period cycle is covered
        min_freq = 1 / (max(time) - min(time)/2)
        max_freq = 1./ 0.5 / (np.median(np.diff(time)))
    else:
        raise ValueError("Definition must be 'Gaia', 'auto', or 'Chloe'")        
        #   ts, y_fit, params, fit_errors = SF.find_timescale(np.array(sf), np.array(t_log), threshold=5e-5, plot=plot, method = 'minuit', last_params=[15, 0.02, 0.08, 1])
        # step_size = 1 / (scale_factor * (last_time - first_time)
    return osf, min_freq, max_freq
    
