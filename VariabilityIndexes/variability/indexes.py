"""
@juliaroquette: 
Class that calculates a series of variability indexes for
a given light-curve.

__This currently includes__
- asymmetry_index (Cody et al. 2014)
- shapiro_wilk
- chi_square
- reduced_chi_square
- iqr
- roms
- anderson_darling
- skewness
- kurtosis
- normalised_excess_variance
- lag1_auto_corr
- von_neumann (abbe)
- norm_ptp
- mad
- periodicity_index

-> Add:
- gaia_AG_proxy

- double check against the Gaia implementation 



__ Under implementation __
- stetsonK

__TO DO__
- fix is_flux flag after it got removed from LightCurve
- Add documenation to each method
- Add references 
- Add Saunders
- Add SL
- Add periodic/aperiodic timescales as an index?


Last update: 02-02-2024
"""
import inspect
import numpy as np
import scipy.stats as ss
from warnings import warn
from variability.lightcurve import LightCurve, FoldedLightCurve
import functools

def min_epochs_property(func):
    """
    Decorator defined to enforce the minimum number of epochs globally
    It returns None if lc.n_epochs < self.min_epochs; 
    else run the function.
    returned values are properties
    """
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        if getattr(self.lc, "n_epochs", 0) < self.min_epochs:
            return None
        return func(self, *args, **kwargs)
    return property(wrapped)

class _tagged_property(property):
    """
    A property that carries a marker for folded-only use.
    This is used to decorate properties of VariabilityIndex that
    should only exists when a FoldedLightCurve is used."""
    def __init__(self, fget=None, *, folded_only=False, **kwargs):
        super().__init__(fget, **kwargs)
        self.folded_only = folded_only

def folded_property(func):
    """
    Decorator for the FoldedLightCurve VariabilityIndexes:
    returns None unless lc is folded
    while also enforcing n_epochs >= min_epochs.
    """
    @functools.wraps(func)
    def _f(self, *args, **kwargs):
        # global min-epochs policy
        if getattr(self.lc, "n_epochs", 0) < self.min_epochs:
            return None
        # folded-only policy
        if not getattr(self, "_is_folded", False):
            return None
        return func(self, *args, **kwargs)
    return _tagged_property(_f, folded_only=True)

class VariabilityIndex:
    _suppress_warnings = False  # class variable to control warning suppression globally
    def __init__(self, lc, min_epochs=5, **kwargs):
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        self.lc = lc
        self._is_folded = isinstance(lc, FoldedLightCurve)
        self._params = kwargs.copy()
        self.min_epochs = int(min_epochs)

    @min_epochs_property
    def std(self):
        """
        Returns the standard deviation of the magnitude values.

        Returns:
            float: Standard deviation.
        """
        return np.std(self.lc.mag,
                    # ddof=1 makes sure std is bias-corrects
                        # this means N-1 is used as the denominator rather than N
                    ddof=1)

    @min_epochs_property
    def weighted_average(self):
        """
        Returns the weighted average of the magnitude values.

        Returns:
            float: Weighted average.
        """
        # this avoids division by zero:
        weights = np.clip(1./(self.lc.err**2), 1e-12, None)
        return np.average(self.lc.mag, weights=weights)

    @min_epochs_property
    def signal_to_noise(self):
        """
        Returns the signal-to-noise ratio of the light curve
        defined as the ratio between the standard deviation of the magnitudes
        and the average uncertainty.

        Returns:
            float: Signal-to-noise ratio.
        """
        return self.std/self.lc.mean_err

    @min_epochs_property
    def shapiro_wilk(self):
        return ss.shapiro(self.lc.mag)[0]

    @folded_property
    def periodicity_index(self):
        return PeriodicityIndex(parent=self).value

    @min_epochs_property
    def asymmetry_index(self):
        # calculate M-index
        M_percentile = self._params.get('M_percentile', 10.)
        M_is_flux = self._params.get('M_is_flux', False)
        return AsymmetryIndex(parent=self,percentile=M_percentile, is_flux=M_is_flux).value

    # @property    
    def Abbe(self):
        """
        Calculate Abbe value as in Mowlavi 2014A%26A...568A..78M
        https://www.aanda.org/articles/aa/full_html/2014/08/aa22648-13/aa22648-13.html
        """
        if self.lc.n_epochs > self.min_epochs:
            return self.lc.n_epochs* np.sum((self.lc.mag[1:] - self.lc.mag[:-1])**2) /\
                2 / np.sum((self.lc.mag - self.lc.mean)**2) / (self.lc.n_epochs- 1)
        else:
            return None

    # this is bugged     
    # @property   
    def stetsonK(self):
        """
        Calculate Stetson index K
        """
        print('odl implementation has a bug')
        return None    
    #     residual = np.sqrt(self.lc.n_epochs)/(self.lc.n_epochs- 1.)*\
    #         (self.mag - self.lc.weighted_average)/self.err
    #     return np.sum(np.fabs(residual)
    #                   )/np.sqrt(self.lc.n_epochs*np.sum(residual**2))


    @min_epochs_property
    def mad(self):
        """
        median absolute deviation
        """
        return ss.median_abs_deviation(self.lc.mag, nan_policy='omit')

    @min_epochs_property
    def chi_square(self):
        """
        Raw Chi-square value
        """
        return np.sum((self.lc.mag - self.weighted_average)**2 / self.lc.err**2)

    @min_epochs_property
    def reduced_chi_square(self):
        """
        Reduced Chi-square value:
        raw chi-square divided by the number of degrees of freedom (N-1)
        """
        return self.chi_square/(np.count_nonzero(
                           ~np.isnan(self.lc.mag)) - 1)
    
    @min_epochs_property
    def iqr(self):
        """
        inter-quartile range
        """
        return ss.iqr(self.lc.mag)
    
    @min_epochs_property
    def roms(self):
        """
        Robust-Median Statistics (RoMS)
        """
        return np.sum(np.abs(self.lc.mag - np.median(self.lc.mag))/self.lc.err)/(self.lc.n_epochs- 1)


    @min_epochs_property
    def normalised_excess_variance(self):
        return (self.std**2 - self.lc.mean_err**2)/self.lc.mean**2

    @min_epochs_property
    def lag1_auto_corr(self):
        if self.lc.n_epochs < self.min_epochs:
            return None
        else:
            return np.sum((self.lc.mag[:-1] - self.lc.mean) *
                      (self.lc.mag[1:] - self.lc.mean))/np.sum(
                          (self.lc.mag - self.lc.mean)**2)
    

    @min_epochs_property
    def norm_ptp(self):
        return (max(self.lc.mag - self.lc.err) - 
            min(self.lc.mag + self.lc.err))/(max(self.lc.mag - self.lc.err) 
                                        + min(self.lc.mag + self.lc.err))    

    @min_epochs_property
    def anderson_darling(self):
        return ss.anderson(self.lc.mag)[0]

    @min_epochs_property
    def skewness(self):
        return ss.skew(self.lc.mag, nan_policy='omit')

    @min_epochs_property
    def kurtosis(self):
        return ss.kurtosis(self.lc.mag)

    @min_epochs_property
    def ptp_5(self):
        """
        Returns the peak-to-peak amplitude of the magnitude values.
        This is defined as the difference between the median values for the datapoints 
        in the 5% outermost tails of the distribution.

        Returns:
            float: Peak-to-peak amplitude.
        """
        return  self.ptp_perc(percentile=5.)

    @min_epochs_property
    def ptp_10(self):
        """
        Returns the peak-to-peak amplitude of the magnitude values.
        This is defined as the difference between the median values for the datapoints 
        in the 10% outermost tails of the distribution.

        Returns:
            float: Peak-to-peak amplitude.
        """
        return  self.ptp_perc(percentile=10.)

    @min_epochs_property
    def ptp_20(self):
        """
        Returns the peak-to-peak amplitude of the magnitude values.
        This is defined as the difference between the median values for the datapoints 
        in the 20% outermost tails of the distribution.

        Returns:
            float: Peak-to-peak amplitude.
        """
        return self.ptp_perc(percentile=20.)

    def ptp_perc(self, percentile=10.):
        """
        Returns the peak-to-peak amplitude of the magnitude values.
        This is defined as the difference between the median values for the datapoints 
        in the `percentile`% outermost tails of the distribution.

        Args:
            percentile (float, optional): Percentile to use for the tails. Defaults to 10..

        Returns:
            float: Peak-to-peak amplitude.
        """
        if (percentile <= 0.) or (percentile >= 49.):
            raise ValueError("Please enter a valid percentile (between 0. and 49.)")
        # it can't get a tail if there are not enough epochs
        if self.lc.n_epochs < 3:
            return None
        tail = int(np.floor(percentile * self.lc.n_epochs / 100.))
        if tail > 1:
            mag_sorted = np.sort(self.lc.mag)
            return  np.median(mag_sorted[-tail:]) - np.median(mag_sorted[:tail])
        else:
            # warn("Not enough epochs to calculate the peak-to-peak amplitude for the given percentile")
            return None

    @folded_property
    def residual_mean(self):
        """
        Average of the residuals of the folded light-curve
        """
        return np.mean(self.lc.residual)
    
    @folded_property
    def residual_std(self):
        """
        Standard deviation of the residuals of the folded light-curve
        """
        return np.std(self.lc.residual, ddof=1)
    
    @folded_property
    def qm_class(self):
        return cody14_q_m_classifier(self.parent.M_index, self.parent.Q_index)
    
    # @folded_property
    # def lafler_kinman(self):
    #     """
    #     Lafler-Kinman string length index of the folded light-curve
    #     """
    #     return np.sum((self.lc.mag_phased[1:] - self.lc.mag_phased[:-1])**2) / np.sum(
    #         (self.lc.mag_phased - self.lc.mean)**2)
    
    # @folded_property
    # def string_length(self):
    #     """
    #     String length index of the folded light-curve
    #     """
    #     return np.sum(np.sqrt(np.diff(self.lc.mag_phased)**2 + np.diff(self.lc.phase)**2))
    
    
    def _list_properties(self):
        """
        This tests if the LightCurve is Folded and return 
        the list of properties accordingly.
        """
        props = []
        is_folded = isinstance(self.lc, FoldedLightCurve)
        for name, value in inspect.getmembers(self.__class__,
                                              lambda o: isinstance(o, property)):
            # Skip folded-only properties if lc is not folded
            if isinstance(value, _tagged_property) and value.folded_only and not is_folded:
                continue
            props.append(name)
        return props
        
    def __str__(self):
        return f'A VariabilityIndex instance has the following properties: {repr(self._list_properties())}'
    
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

class AsymmetryIndex:
    def __init__(self, parent, percentile=10., is_flux=False):
        self.parent = parent
        self._percentile = float(percentile)
        self.is_flux = bool(is_flux)

    @property
    def percentile(self):
        """ 
        Percentile used to calculate the M-index 
        """
        return self._percentile

    @percentile.setter
    def percentile(self, new_percentile):
        if (new_percentile > 0) and (new_percentile < 49.):
            self._percentile = new_percentile
        else:
            raise ValueError("Please enter a valid percentile (between 0. and 49.)")

    @property
    def get_percentile_mask(self):
        return (self.parent.lc.mag <= \
                            np.percentile(self.parent.lc.mag, self._percentile))\
                            | (self.parent.lc.mag >= \
                                np.percentile(self.parent.lc.mag, 100 - self._percentile))
                            
    @property
    def value(self):
        return (1. - 2*int(self.is_flux))*(np.mean(self.parent.lc.mag[self.get_percentile_mask]) - self.parent.lc.median)/self.parent.std

class PeriodicityIndex:
    def __init__(self, parent):
        self.parent = parent

    @property
    def value(self):
        # print(self._waveform_type)
        # self.parent.lc._get_waveform()
        return (np.std(self.parent.lc.residual, ddof=1)**2 - np.mean(self.parent.lc.err_phased)**2)\
            /(np.std(self.parent.lc.mag_phased, ddof=1)**2 - np.mean(self.parent.lc.err_phased)**2)


def cody14_q_m_classifier(M_index, Q_index):
    """
    Simple classifier based on M and Q indices
    based on Cody et al. 2014 (https://ui.adsabs.harvard.edu/#abs/2014AJ....147...82C)
    """
    if M_index is None or Q_index is None:
        return None
    elif (M_index > 0.25) and (0 <= Q_index < 0.11):
        return 'EB'
    elif (M_index <= 0.25) and (0 <= Q_index < 0.11):
        return 'P'
    elif (abs(M_index) <= 0.25) and (0.11 <= Q_index <= 0.61):
        return 'QPS'
    elif (M_index > 0.25) and (0.11 <= Q_index <= 0.61):
        return 'QPD'
    elif (M_index > 0.25) and (1 <= Q_index > 0.61):
        return 'APD'
    elif (M_index < -0.25) and (Q_index > 0.11):
        return 'B'
    elif (abs(M_index) <= 0.25) and (1 <= Q_index > 0.61):
        return 'S'
    else:
        return 'Unclassified'


def gaia_AG_proxy(phot_g_mean_flux, phot_g_mean_flux_error, phot_g_n_obs):
    """
    Following Mowlavi et al. 2021 (https://ui.adsabs.harvard.edu/#abs/2021A%26A...648A..44M)
    this function calculates a proxy for the variability of Gaia sources using the uncertainty of
    the Gaia G-band fluxes.
    This is provided in equation (2) of the paper, and is given by:
    
    AG = sqrt(phot_g_n_obs)*phot_g_mean_flux_error/phot_g_mean_flux

    For constant stars this is approximately the standard deviation of 
    G light curves due to noise and uncalibrated systematic effects. 
    _summary_

    Args:
        phot_g_mean_flux (_type_): _description_
        phot_g_mean_flux_error (_type_): _description_
        phot_g_n_obs (_type_): _description_
    """
    return np.sqrt(phot_g_n_obs)*phot_g_mean_flux_error/phot_g_mean_flux