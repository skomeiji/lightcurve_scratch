"""
Colection of fitlers
"""

import warnings
import numpy as np
import scipy as sp
import pandas as pd
from astropy.convolution import Box1DKernel, convolve

class Filtering:
    def __init__(self, lc):
        from variability.lightcurve import LightCurve
        if isinstance(lc, LightCurve):
            self.lc = lc
        else:
            raise TypeError("lc must be an instance of LightCurve")
        
        self.even = self.is_evenly_spaced()
        # if not self.even:
            # warnings.warn("Time series may not be evenly spaced.", UserWarning)

    def filter(self, method=None, **kwargs):
        """
        Apply a filter of choice to detrend light-curve.

        Parameters:
        - method (str): The filtering method to be applied. Available options are:
            - 'savgol': Apply the Savitzky-Golay filter to remove long-term trends.
                - **kwargs:
                    - timescale (int): The length of the filter window.
                    - polyorder (int): The order of the polynomial used to fit the samples.
            - 'Cody': Apply the Cody filter to remove long-term trends.
                - **kwargs:
                    - window_length (int): The length of the filter window.
                    - polyorder (int): The order of the polynomial used to fit the samples.
            - 'rolling_average': Apply the rolling average filter to remove long-term trends.
                - **kwargs:
                    - window_size (int): The size of the moving window.
            - 'uneven_savgol': Apply the uneven Savitzky-Golay filter to remove long-term trends.
                - **kwargs:
                    - breakpoints (list): A list of breakpoints where the filter window changes.
                    - window_length (int): The length of the filter window for each segment.
                    - polyorder (int): The order of the polynomial used to fit the samples for each segment.
            - 'smooth_per_timescale': Apply the smooth per timescale filter to remove long-term trends.
                - **kwargs:
                    - timescale (float): The timescale parameter for the filter.

        Returns:
        - The filtered light-curve.

        Raises:
        - ValueError: If the specified filtering method is not implemented.
        """
        if method == 'savgol':
            return self.savgol_longtrend(**kwargs)
        elif method == 'Cody':
            return self.Cody_long_trend(**kwargs)
        elif method == 'rolling_average':
            return self.rolling_average(**kwargs)
        elif method == 'uneven_savgol':
            return self.uneven_savgol(**kwargs)
        elif method == 'smooth_per_timescale':
            return self.smooth_per_timescale(**kwargs)
        else:
            raise ValueError("Method _{0}_ not implemented.".format(method))
    
    def is_evenly_spaced(self, tolerance=0.1):
        """
        tests if a function is close to evenly spaced.
        The default tolerance check if less than 0.01% of t
        """
        dt = self.lc.time[1:] - self.lc.time[:-1]
        num_outliers = len(np.where(abs(dt - np.mean(dt)) > 3*np.std(dt, ddof=1))[0])
        return bool(num_outliers < (tolerance/100.)*len(self.lc.time))

    #masking
    def sigma_clip(self, sigma=5):
        """
        remove outliers from the light-curve using a sigma clipping
        returns as mask for values to be kept after filtering
        """
        return np.abs(self.lc.mag - self.lc.median) <= sigma * self.lc.std

    def get_num_time(self, timescale=10.):
        """
        Assuming the data is in days, this 
        function conts how many points you need 
        to cover a given timescale
        """
        if not self.even:
            warnings.warn("Time series may not be evenly spaced - no direct conversion between number of points and timescale.", UserWarning)
        time = self.lc.time - np.min(self.lc.time)
        return len(time[time < timescale])
    
    #detrending
    def Cody_long_trend(self, timescale=10.):
        """
        Code used in Cody et al. 2018,2020
        window=715 is appropriate for K2 data and equivalent to 10days
        Source:
        """
        window = self.get_num_time(timescale)
        return scipy.ndimage.median_filter(self.lc.mag, size=window, mode='nearest')
    
        
    def savgol_longtrend(self, timescale=10., polyorder=3) :
        """
        To be changed to timescale rather than window size
        """
        if not self.even:
            warnings.warn("SavGol Scipy expects even time series - which may not be true here.", UserWarning)        
        window = self.get_num_time(timescale)
        return sp.signal.savgol_filter(self.lc.mag, window, polyorder)
    
    def uneven_savgol(self, window, polyorder):
        x = self.lc.time
        y = self.lc.mag
        return uneven_savgol(x, y, window, polyorder)
    
    def smooth_per_timescale(self, window_days=10.):
        """
        for given light-curve with a time and mag, 
        use a window wd in days to smooth the light-curve
        then remove the smoothed curve form the 
        """
        smoothed_values = np.zeros_like(self.lc.mag, dtype=float)
        for i in range(len(self.lc.time)):
            start_time = self.lc.time[i] - window_days/2
            end_time = self.lc.time[i] + window_days/2
            select = np.where((self.lc.time >= start_time) & (self.lc.time <= end_time))
            smoothed_values[i] = np.mean(self.lc.mag[select])
        return smoothed_values

    def rolling_average(self, window=5):
        """
        Uses
        """
        if not self.even:
            warnings.warn("Using window size in datapoints number for uneven data.", UserWarning)                
        return pd.Series(self.lc.mag).rolling(window, min_periods=window, win_type='boxcar', center=True, closed='neither').mean().to_numpy()
    
    def apply_filter(self, mask):
        """
        applies a mask generated by a filter into a light curve
        """
        from variability.lightcurve import FoldedLightCurve
        self.lc.time = np.asarray(self.lc.time, dtype=float)[mask]
        self.lc.mag = np.asarray(self.lc.mag, dtype=float)[mask]
        self.lc.err = np.asarray(self.lc.err, dtype=float)[mask]
        if isinstance(self, FoldedLightCurve):
            # if the light curve is folded, we need to update the phase
            self.lc._get_phased_values()
            self.lc.wf = WaveForm(self.lc.phase, self.lc.mag_phased)
            self.lc._get_waveform()
        return self.lc
        


class WaveForm:
    def __init__(self, phase, mag_phased):
        """
        Initialize the WaveForm class.

        Parameters:
        - phase (np.array): The phase of the folded light curve.
        - mag_phased (np.array): The magnitude of the folded light curve.

        Raises:
        - TypeError: If folded_lc is not an instance of FoldedLightCurve.
        """
        self.phase = phase
        self.mag_phased = mag_phased
        self.N = len(self.mag_phased)

    def circular_rolling_average_number(self, window_size=5):
        """
        Calculate the circular rolling average of the folded light curve.

        Parameters:
        - window_size (int): The window size for the rolling average.

        Returns:
        - np.array: The circular rolling average waveform.
        """
        # makes sure it won't break with weird sizes of lc
        if not isinstance(window_size, int):
            raise TypeError("window_size must be an integer")
        if window_size <= 0 or window_size >= self.N:
            raise ValueError("window_size must be > 0 and smaller than the data length")
        if window_size % 2 == 0:
            raise ValueError('The "window" must be an odd integer')

        #
        half_window = window_size //2
        # to deal with edges, I am assuming phased light curves are circular
        extended_mag = np.concatenate((self.mag_phased, self.mag_phased, self.mag_phased))
        #
        extended_waveform = np.array([
            np.mean(extended_mag[i - half_window : i + half_window + 1])\
                    for i in np.arange(half_window, 
                                       len(extended_mag) - half_window)])
        
        return extended_waveform[self.N - half_window: 2 * self.N - half_window]
        
    def savgol(self, window=10, polyorder=3):
        return sp.signal.savgol_filter(self.mag_phased, window, polyorder)

    def circular_rolling_average_phase(self, wd_phase=0.1):
        """
        Calculate the circular rolling average of the folded light curve based on phase.

        Parameters:
        - wd_phase (float): The phase window size for the rolling average.

        Returns:
        - np.array: The circular rolling average waveform based on phase.
        """
        waveform = np.full(len(self.phase), np.nan)
        extended_phase = np.concatenate((self.phase - 1, self.phase, 1 + self.phase))
        extended_mag = np.concatenate((self.mag_phased, self.mag_phased, self.mag_phased))
        for i, p in enumerate(self.phase):
            select = np.where((extended_phase <= p + wd_phase/2.) & (extended_phase > p - wd_phase/2.))[0]
            waveform[i] = np.mean(extended_mag[select])
        return waveform

    def waveform_H22(self, kernel=4):
        """
        Code used in Hillenbrand et al. 2022:
        Source: https://github.com/HarritonResearchLab/NAPYSOs
        """
        # Create the residual curve
        # We use three periods and extract the middle to prevent edge effects
        three_periods = np.concatenate((self.mag_phased, self.mag_phased, self.mag_phased))
        boxcar = Box1DKernel(len(self.mag_phased) // kernel)
        smooth_mag = convolve(three_periods, boxcar)
        smooth_mag = smooth_mag[np.size(self.mag_phased):2*np.size(self.mag_phased)]
        return smooth_mag

    def waveform_Cody(self, n_point=50):
        return sp.ndimage.median_filter(self.mag_phased, size=n_point, mode='wrap')   
    
    def uneven_savgol(self, window, polyorder):
        """
        Apply a Savitzky-Golay filter to the folded light curve with non-uniform spacing.

        Parameters:
        - window (int): Window length of datapoints. Must be odd and smaller than the light curve size.
        - polyorder (int): The order of polynomial used. Must be smaller than the window size.

        Returns:
        - np.array: The smoothed folded light curve.

        Raises:
        - ValueError: If the data size is smaller than the window size or if the window is not an odd integer.
        - TypeError: If window or polynom are not integers.
        """
        x = np.concatenate((self.phase - 1., self.phase, 1. + self.phase))
        y = np.concatenate((self.mag_phased, self.mag_phased, self.mag_phased)) 
        # print(f"Debugging uneven_savgol with window={window}, polyorder={polyorder}")
        return  uneven_savgol(x, y, window, polyorder)  [self.N:2*self.N]
    
    def get_waveform(self, waveform_type='uneven_savgol', waveform_params={}):
        if waveform_type == 'savgol':
            window = waveform_params.get('window', 25.)
            polyorder = waveform_params.get('polyorder', 3)
            waveform = self.savgol(window=window, polyorder=polyorder)
        elif waveform_type == 'Cody':
            n_point = waveform_params.get('n_point',50)
            waveform = self.waveform_Cody(n_point=n_point)
        elif waveform_type == 'circular_rolling_average_phase':
            wd_phase = waveform_params.get('wd_phase', 0.25)
            waveform = self.circular_rolling_average_phase(wd_phase=wd_phase)
        elif waveform_type == 'circular_rolling_average_number':
            window_size = waveform_params.get('window_size', 0.1*self.N)
            window_size = int(round(window_size))
            waveform = self.circular_rolling_average_number(window_size=window_size)
        elif waveform_type == 'H22':
            kernel = waveform_params.get('kernel', 4.)
            waveform = self.waveform_H22(kernel=kernel)
        elif waveform_type == 'uneven_savgol':
            window = waveform_params.get('window', 0.25*self.N)
            window = int(round(window))
            if window % 2 == 0:
                window += 1
            polyorder = waveform_params.get('polyorder', 1)
            while polyorder >= window:
                window += 2
            waveform = self.uneven_savgol(window, polyorder)
        else:
            raise ValueError(f"Method '{waveform_type}' not implemented.")
        return waveform
    
    def residual_magnitude(self, waveform):
        """
        Calculate the residual magnitude after waveform subtraction.
        """
        return self.mag_phased - waveform

# def uneven_savgol_(x, y, window, polyorder):
#     """
#     Applies a Savitzky-Golay filter to y with non-uniform spacing
#     as defined in x

#     This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
#     The borders are interpolated like scipy.signal.savgol_filter would do

#     Parameters
#     ----------
#     window : int (odd)
#         Window length of datapoints. Must be odd and smaller than x
#     polyorder : int
#         The order of polynom used. Must be smaller than the window size

#     Returns
#     -------
#     np.array of float
#         The smoothed y values
#     """
#     if len(x) != len(y):
#         raise ValueError('"x" and "y" must be of the same size')
#     if len(x) < window:
#         raise ValueError('The data size must be larger than the window size')

#     if type(window) is not int:
#         raise TypeError('"window" must be an integer')

#     if window % 2 == 0:
#         raise ValueError('The "window" must be an odd integer')

#     if type(polyorder) is not int:
#         raise TypeError('"polyorder" must be an integer')

#     if polyorder >= window:
#         raise ValueError('"polyorder" must be less than "window"')

#     half_window = window // 2
#     polyorder += 1

#     # Initialize variables
#     A = np.empty((window, polyorder))     # Matrix
#     tA = np.empty((polyorder, window))    # Transposed matrix
#     t = np.empty(window)                # Local x variables
#     y_smoothed = np.full(len(y), np.nan)

#     # Start smoothing
#     for i in range(half_window, len(x) - half_window, 1):
#         # Center a window of x values on x[i]
#         for j in range(0, window, 1):
#             t[j] = x[i + j - half_window] - x[i]

#         # Create the initial matrix A and its transposed form tA
#         for j in range(0, window, 1):
#             r = 1.0
#             for k in range(0, polyorder, 1):
#                 A[j, k] = r
#                 tA[k, j] = r
#                 r *= t[j]

#         # Multiply the two matrices
#         tAA = np.matmul(tA, A)

#         # Invert the product of the matrices
#         tAA = np.linalg.inv(tAA)

#         # Calculate the pseudoinverse of the design matrix
#         coeffs = np.matmul(tAA, tA)

#         # Calculate c0 which is also the y value for y[i]
#         y_smoothed[i] = 0
#         for j in range(0, window, 1):
#             y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

#         # If at the end or beginning, store all coefficients for the polyorder
#         if i == half_window:
#             first_coeffs = np.zeros(polyorder)
#             for j in range(0, window, 1):
#                 for k in range(polyorder):
#                     first_coeffs[k] += coeffs[k, j] * y[j]
#         elif i == len(x) - half_window - 1:
#             last_coeffs = np.zeros(polyorder)
#             for j in range(0, window, 1):
#                 for k in range(polyorder):
#                     last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

#     # Interpolate the result at the left border
#     for i in range(0, half_window, 1):
#         y_smoothed[i] = 0
#         x_i = 1
#         for j in range(0, polyorder, 1):
#             y_smoothed[i] += first_coeffs[j] * x_i
#             x_i *= x[i] - x[half_window]

#     # Interpolate the result at the right border
#     for i in range(len(x) - half_window, len(x), 1):
#         y_smoothed[i] = 0
#         x_i = 1
#         for j in range(0, polyorder, 1):
#             y_smoothed[i] += last_coeffs[j] * x_i
#             x_i *= x[i] - x[-half_window - 1]

#     return y_smoothed

def uneven_savgol(x, y, window, polyorder):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    Parameters
    ----------
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polyorder : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')
    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if not isinstance(window, int):
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if not isinstance(polyorder, int):
        raise TypeError('"polyorder" must be an integer')

    if polyorder >= window:
        raise ValueError('"polyorder" must be less than "window"')

    half_window = window // 2
    polyorder += 1
    
    # Initialize variables
    t = np.empty(window)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Calculate powers of t
    powers_of_t = np.empty((window, polyorder))
    for j in range(window):
        powers_of_t[j] = np.power(x[half_window + j] - x[half_window], np.arange(polyorder))

    # Start smoothing
    for i in range(half_window, len(x) - half_window):
        # Calculate t vector
        t = x[i - half_window:i + half_window + 1] - x[i]

        # Create the initial matrix A and its transposed form tA
        A = np.vander(t, polyorder, increasing=True)
        tA = A.T

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA_inv = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA_inv, tA)

        # Calculate smoothed y value
        y_smoothed[i] = np.dot(coeffs[0], y[i - half_window:i + half_window + 1])

    return y_smoothed