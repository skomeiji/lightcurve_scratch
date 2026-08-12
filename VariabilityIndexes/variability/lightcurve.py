"""
Base class for reading light curves

@juliaroquette:
Defines a base class for reading light curves, which can be inherited by other classes that analyse it.

Last update: 11 July 2025
Noted an inconsistency between timescale and waveform derivation between FoldedLightCurve, VariabilityIndex and and TimeScale.

previous update: 
- 30 June 2025
Carried out some debugging
- 31 January 2024: 
Last version

TO DO:
- ptp and ramge are now repeated properties, should be fixed
- Add metod for getting light curve time properties 
- Define a multi-wavelength light curve class
- Define a class/function that generates a sample of light-curves


To add:

typical dt
minimum dt
maximum dt

minimum dphase
mean dphase
maximum dphase
"""

import numpy as np
import scipy.stats as ss
import pandas as pd
import inspect
import warnings
np.random.seed(42)
from variability.filtering import WaveForm, Filtering
 
class LightCurve:
    """
    Class representing a light curve. 

    Attributes:
        time (ndarray): Array of time values.
        mag (ndarray): Array of magnitude values.
        err (ndarray): Array of magnitude error values.
    
    Properties:
        - n_epochs (int): Number of datapoints in the light curve.
        - time_span (float): Total time-span of the light curve.
        - mean (float): Mean of the magnitude values.
        - weighted_average (float): Weighted average of the magnitude values.
        - median (float): Median of the magnitude values.
        - min (float): Minimum value of the magnitudes.
        - max (float): Maximum value of the magnitudes.
        - time_max (float): Maximum value of the observation times.
        - time_min (float): Minimum value of the observation times.
        - ptp (float): range of magnitude values (peak-to-peak amplitude).
    """
    def __new__(cls, *, time, mag, err):
        """
        Tests if the minimum requirements to create a LightCurve object are met.
        """
        # check the shape
        if len(time) != len(mag) or len(mag) != len(err):
            raise ValueError("time, mag, and err must all have the same shape")
        # check that there is at least one finite value in time, mag and err
        finite = np.isfinite(time) & np.isfinite(mag) & np.isfinite(err)
        if not finite.any():
            # abort construction → caller gets None
            return None
        # otherwise proceed with normal instance creation
        return super().__new__(cls)

    def __init__(self, *, time, mag, err):
        """
        Initializes a LightCurve object.

        Args:
            time (ndarray): Array of time values.
            mag (ndarray): Array of magnitude values.
            err (ndarray): Array of error values.
        """
        # check time, mag, err are all valid
        mask = np.isfinite(time) & np.isfinite(mag) & np.isfinite(err)
        self.time = np.asarray(time, dtype=float)[mask]
        self.mag = np.asarray(mag, dtype=float)[mask]
        self.err = np.asarray(err, dtype=float)[mask]
        #
        # makes sure light curves are sorted by time
        #
        sorted_indices = np.argsort(self.time)
        self.time = self.time[sorted_indices]
        self.mag = self.mag[sorted_indices]
        self.err = self.err[sorted_indices]
        self.dt = np.diff(self.time)

    @property    
    def n_epochs(self):
        """
        Returns the number of datapoints in the light curve.

        Returns:
            int: Number of datapoints.
        """
        return int(len(self.mag))
    
    @property
    def time_span(self):
        """
        Returns the total time-span of the light-curve

        Returns:
            float: light-curve time-span.
        """
        return np.max(self.time) - np.min(self.time)
    
    @property
    def mean(self):
        """
        Returns the mean of the magnitude values.

        Returns:
            float: Mean value.
        """
        return self.mag.mean()
    
    @property
    def mean_err(self):
        """
        Returns the mean of the uncertainty values.

        Returns:
            float: Mean value.
        """
        return self.err.mean()    
    
    @property
    def max(self):
        """
        Returns the max value of the magnitudes.

        Returns:
            float: mags value.
        """
        return self.mag.max()
    
    @property
    def min(self):
        """
        Returns the min value of the magnitudes.

        Returns:
            float: max mags value.
        """
        return self.mag.min()
    
    @property
    def time_max(self):
        """
        Returns the max value of the observations times.

        Returns:
            float: max time value.
        """
        return self.time.max()
    
    @property
    def time_min(self):
        """
        Returns the min value of the observations times.

        Returns:
            float: min time value.
        """
        return self.time.min()

    @property
    def median(self):
        """
        Returns the median of the magnitude values.

        Returns:
            float: Median value.
        """
        return np.median(self.mag)
    
    @property
    def range(self):
        """
        Returns the (range) peak-to-peak amplitude of the magnitude values.
        This follows a simple definition of peak-to-peak amplitude as the difference between the maximum and minimum values.

        Returns:
            float: (range) peak-to-peak amplitude.
        """
        # range can only be zero if all values are the same        
        range = np.max(self.mag) - np.min(self.mag)
        if len(self.mag) > 1:
            return range
        else:
            return None

    @property
    def dt_median(self):
        return np.median(self.dt)

    @property
    def dt_min(self):
        return np.min(self.dt)

    @property
    def dt_max(self):
        return np.max(self.dt)

    @property
    def dt_10th(self):
        return np.percentile(self.dt, 10)
    
    @property
    def dt_90th(self):
        return np.percentile(self.dt, 90)
    
    @property
    def dt_geomean(self):
        return ss.gmean(self.dt)
    
    
    # # @property
    # def ():
    #     pass
    
        
    def get_timescale_properties(self):
        # get the difference between consecutive time values
        
        pass
    
    def _list_properties(self):
        """
        list properties of the class LightCurve
        """
        property_names = [name for name, value in inspect.getmembers(self.__class__, lambda o: isinstance(o, property))]
        return property_names    
        
    def __str__(self):
        return f'A LightCurve instance has the following properties: {repr(self._list_properties())}'
    
    def __len__(self):
        return self.n_epochs


class FoldedLightCurve(LightCurve):
    """
    Represents a folded light curve with time values folded for a given timescale.
    """
    # Control for warnings
    _suppress_warnings = False
    def __new__(cls, **kwargs):
        if 'lc' in kwargs:
            if not isinstance(kwargs['lc'], LightCurve):
                raise TypeError("'lc' must be a LightCurve instance")
            if all(key in kwargs for key in ['time', 'mag', 'err']):
                warnings.warn("Both 'lc' and 'time', 'mag', 'err' were, 'lc' will be used", UserWarning)
            return super().__new__(cls, time=kwargs['lc'].time, mag=kwargs['lc'].mag, err=kwargs['lc'].err)
        # otherwise, test is time, mag, err needed to build LightCurve were passed
        elif all(key in kwargs for key in ['time', 'mag', 'err']):
            return super().__new__(cls, time=kwargs['time'], mag=kwargs['mag'], err=kwargs['err'])
        else:
            raise ValueError("Either a lc=LightCurve object or time, mag and err arrays must be provided")

    def __init__(self, *,
                 timescale,
                 **kwargs):
        if 'lc' in kwargs:
            # this guarantees that the time, mag, err is the same in both
            # the current and the parent object. This is, with lc=LightCurve(...)
            # and lc_f = FoldedLightCurve(lc=lc), lc_f.time is lc.time, and so on
            # self.time = kwargs['lc'].time
            # self.mag  = kwargs['lc'].mag
            # self.err  = kwargs['lc'].err
            # this was problematic with transmiting the properties, so I changed back for now
            super().__init__(time=kwargs['lc'].time,
                             mag=kwargs['lc'].mag,
                             err=kwargs['lc'].err)            
        elif all(key in kwargs for key in ['time', 'mag', 'err']):
            super().__init__(time=kwargs['time'], mag=kwargs['mag'], err=kwargs['err'])
        else:
            raise ValueError("Either a LightCurve object or time, mag and err arrays must be provided")

        # FoldedLightCurve needs a timescale
        if (not isinstance(timescale, (int, float))) or (timescale <= 0.):
            raise ValueError("FoldedLightCurve object requires a timescale (positive float/int value) to be defined.")
        else:
            self._timescale = timescale
        # Now phasefold the light curve to the given timescale
        self._reference_time = kwargs.get('reference_time', 0.)
        self._get_phased_values()        
                
        # phasefold lightcurves have a waveform
        self._waveform_type = kwargs.get('waveform_type', 'uneven_savgol')
        self._waveform_params = kwargs.get('waveform_params', {'window': round(.25*self.n_epochs), 'polyorder': 2})
        self._get_waveform()
        # estimate phase differences
        # this if required for several properties
        self._phase_diffs = np.diff(self.phase)


    def _get_phased_values(self):
        # _reference_time adds an offset to the phase 
        # this may be useful to align phases to a specific event. 
        # Calculate the phase values
        phase = np.mod(self.time - self._reference_time, self._timescale) / self._timescale
        # Sort the phase and magnitude arrays based on phase values
        phase_number = np.floor((self.time - self._reference_time)/self._timescale)
        sort = np.argsort(phase)
        self.phase = phase[sort]
        self.mag_phased = self.mag[sort]      
        self.err_phased = self.err[sort]
        self.time_phased = self.time[sort]
        self.phase_number = phase_number[sort]

        
    @property
    def timescale(self):
        return self._timescale    
    
    @timescale.setter
    def timescale(self, new_timescale):
        if (not isinstance(new_timescale, (int, float))) or (new_timescale <= 0.):
            raise ValueError("timescale must be a positive float/int value.")
        else:
            self._timescale = new_timescale
            # update phase-folded values
            self._get_phased_values()
            # update the waveform for the new timescale
            self._get_waveform()
    
    @property
    def reference_time(self):
        """ Reference time for phase folding."""
        return self._reference_time

    @reference_time.setter
    def reference_time(self, new_reference_time):
        self._reference_time = new_reference_time
        self._get_phased_values()
        self._get_waveform()
    
    @property
    def n_cycle(self):
        return np.nanmax(self.phase_number)
    
    @property
    def waveform_type(self):
        return self._waveform_type    
    
    @waveform_type.setter
    def waveform_type(self, new_type):
        self._waveform_type = new_type
        self._get_waveform()

    @property
    def waveform_params(self):
        return self._waveform_params

    @property
    def max_phase_gap(self):
        """
        Maximum gap in phase coverage
        """
        if self.n_epochs < 2:
            return None
        return np.max(self._phase_diffs)
    
    @property
    def saunders_norm(self):
        """
        Estimaste the Saunders diagnosis for time coverage of 
        a time series
        Saunders et al. (2006, Astronomische Nachrichten, 327, 783)

        Normalized Saunders statistic:
        - 0 = perfectly uniform phase coverage
        - ~1 = random uniform coverage
        - >1 = clumpy coverage
        """
        if self.n_epochs < 2:
            return np.nan

        # Sort phases and compute gaps with wrap-around
        S = self.n_epochs * np.sum(np.diff(self.phase, append=self.phase[0] + 1.0)**2)

        # Normalize
        S_exp = (2.0 * self.n_epochs) / (self.n_epochs + 1.0)
        return (S - 1.0) / (S_exp - 1.0)

    @waveform_params.setter
    def waveform_params(self, new_params):
        self._waveform_params = new_params
        self._get_waveform()   
        
    def _get_waveform(self):
        """
        Derives the waveform and update the residual curve
        """
        wf = WaveForm(self.phase, self.mag_phased)
        self.waveform = wf.get_waveform(
            waveform_type=self._waveform_type,
            waveform_params=self._waveform_params
        )
        # phasefolded lightcurves also have a residual curve between the waveform and the lightcurve
        self.residual = wf.residual_magnitude(self.waveform)

    def set_waveform(self, waveform_type=None, waveform_params=None):
        """
        Update both waveform_type and waveform_params at the same time
        """
        if waveform_type is not None:
            self._waveform_type = waveform_type
        if waveform_params is not None:
            self._waveform_params = waveform_params
        self._get_waveform()
    
    def refold(self, *, timescale=None, reference_time=None,
            waveform_type=None, waveform_params=None):
        changed_phase = False
        if timescale is not None:
            if not isinstance(timescale, (int, float)) or timescale <= 0:
                raise ValueError("timescale must be a positive number")
            self._timescale = float(timescale); changed_phase = True
        if reference_time is not None:
            if not isinstance(reference_time, (int, float)):
                raise ValueError("reference_time must be numeric")
            self._reference_time = float(reference_time); changed_phase = True
        if changed_phase:
            self._get_phased_values()
        if waveform_type is not None:
            self._waveform_type = waveform_type
        if waveform_params is not None:
            self._waveform_params = waveform_params
        self._get_waveform()
    
    def _list_properties(self):
        """
        List properties of the class, excluding 'waveform_params'.
        """
        property_names = [
            name for name, value in inspect.getmembers(self.__class__, lambda o: isinstance(o, property))
            if name != 'waveform_params'
        ]
        return property_names
        
    def __str__(self):
        return f'A FoldedLightCurve instance has the following properties: {repr(self._list_properties())}'

    @classmethod
    def suppress_warnings_globally(cls):
        """
        This is class method that enable to suppress warnings globally
        for FoldedLightCurve instances.
        
        usage:
        FoldedLightCurve.suppress_warnings_globally()
        """
        cls._suppress_warnings = True

    @classmethod
    def enable_warnings_globally(cls):
        """
        This is a class method to enable
        warnings globally for FoldedLightCurve instances.
        Usage:
        FoldedLightCurve.enable_warnings_globally()
        """
        cls._suppress_warnings = False        

    def __repr__(self):
        return (f"<FoldedLightCurve(timescale={self._timescale}, "
                f"reference_time={self._reference_time}, "
                f"waveform_type={self._waveform_type}, "
                f"N={self.n_epochs})>")

class DetrendedLightCurve:
    """
    A LightCurve subclass that stores both the raw and detrended magnitudes.

    Parameters
    ----------
    lc : LightCurve
        Input light curve object to detrend.
    timescale : float
        Smoothing timescale in days used for long-term trend removal.
    method : str, optional
        The smoothing method to use. Default: 'smooth_per_timescale'
    kwargs :
        Additional keyword arguments passed to the chosen filter (e.g., window_days).

    Attributes
    ----------
    raw_mag : np.ndarray
        Original (unmodified) magnitudes from the input LightCurve.
    mag : np.ndarray
        Detrended magnitudes (original minus smoothed trend).
    smooth_mag : np.ndarray
        The smoothed (trend) version of the light curve.
    even : bool
        Whether the input light curve is evenly spaced.
    """

    def __init__(self, lc, timescale, method='smooth_per_timescale', **kwargs):

        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")

        # Copy all LightCurve attributes into self
        self.__dict__.update(lc.__dict__.copy())

        # Store original magnitudes
        self.raw_mag = np.copy(lc.mag)

        # Initialize filter
        filtering = Filtering(lc)
        self.even = filtering.even

        # Choose detrending method
        if method != 'smooth_per_timescale':
            warnings.warn(
                f"Currently, only 'smooth_per_timescale' is supported for DetrendedLightCurve. "
                f"'{method}' will be ignored and 'smooth_per_timescale' used instead.",
                UserWarning
            )

        # Compute smoothed (trend) component
        self.smooth_mag = filtering.smooth_per_timescale(window_days=timescale, **kwargs)

        # Replace magnitudes by detrended ones
        self.mag = self.raw_mag - self.smooth_mag

        # Optionally, keep timescale info
        self.timescale = timescale
        self.method = method

    def __repr__(self):
        cls = self.__class__.__name__
        npts = len(self.time) if hasattr(self, "time") else "?"
        return f"<{cls}: {npts} points, timescale={self.timescale} days>"

    def get_trend(self):
        """Return the smoothed (trend) magnitude curve."""
        return self.smooth_mag

    def get_detrended_mag(self):
        """Return the detrended magnitudes (raw - trend)."""
        return self.mag

    def get_raw_mag(self):
        """Return the original unmodified magnitudes."""
        return self.raw_mag
