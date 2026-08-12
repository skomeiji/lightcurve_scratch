# A Python package for deriving variability features from any kind of light-curves

**@juliaroquette** Package under development for variability indexes for any time type of light-curves. 


**Last Update**: (18th September 2025) improved both implementation and documentation of variability indexes.

In this current version, one can import and use it by doing:

```python
import sys
sys.path.append('PAT/TO/THE/PACKAGE/LOCATION')  
```

TO DO:
<details>
- :white_large_square: Proper packaging
- :white_large_square: Proper installation tutorial
</details>

# `lightcurve` module:

Provides tools for loading light curves as objects. Three distinct classes related to light curves are currently included: 
- `LightCurve`s are the simplest light curves
- `FoldedLightCurve`s are phase-folded light curves with a known timescale
- `SyntheticLightCurve` (under construction) provide a suite of models of light curves for different variability modes and survey fingerprints. 
- `MultiBandLighCurve`  (not implemented yet) deals with (quasi-)simultaneous multiband light curves.

Throughout this documentation, we approach an observed light-curve as a series of observations, $\{m_i\}$, where:

$$m_i = m(t_i) + a(t_i) + \epsilon_{i},$$

where $m_i$ is the $i$-th observation at the time $t_i$, $\{m(t_i)\}$ are a series of snapshots of a time-dependent signal at the time $t_i$ (or the primary signal in the light-curve).  $\{a(t_i)\}$ is any secondary signal, which is often assume to be zero. $\epsilon_{i}$ is the photometric uncertainty for the $i$-th observation, which in an ideal case is not too far from the survey's typical uncertainty, $\bar{\epsilon}$.

For example, let's consider the following light-curve:

````python
import numpy as np
N = 100 
time = np.linspace(0, 80, N)
err =  0.01 * np.random.random_sample(N)
period = 10. 
amplitude = 0.5
noise =  np.random.normal(scale=0.05, size=N)
mag_sin = 0.5 * amplitude * np.sin(2 * np.pi * time / period) + noise
````
**Note:** it is important to note that astronomical magnitudes are obtained from the observed flux of stars. For the fluxes, it is common to assume that errors are Gaussian distributed, which is ok for bright sources (but not always true!). However, when converting fluxes to magnitudes, $m_i=-2.5\log\big(\frac{f_i}{f_0}\big)$, uncertainty propagation yield asymmetric uncertainties. It is a common assumption that error are symmetric even in magnitude, with the argument that that for small errors in flux, a log function will be approximately linear. This note must be kept in mind for both when simulating light curves, but also when working with variability indexes that assume a specific behavior for the case of constant stars dominated by gaussian noise. 

## `LightCurve` class

 `LightCurve` is a class that formats input time-series data, $\{t_i, m_i, \epsilon_i\}$, into a standard representation. 

To instantiate a `LightCurve` object:

```python
  from variability.lightcurve import LightCurve
  lc = LightCurve(time, mag, err)
```

Where the attributes `time`, `mag`, and `err` are numpy-arrays with the same length providing the observational time, magnitudes and magnitude uncertainties respectively. 

Additionally to the time-series itself, `LightCurve` objects have a series of descriptive properties attached to itself. The list of properties currently implemented, can be accessed using:

```python
lc._list_properties()
```

- `n_epochs` : Number of datapoints (number of epochs), $N$, in the light curve.


- `mean` : simple average of the magnitudes.

$$\bar{m} = \frac{1}{N} \sum_{i=1}^{N} m_i$$

- `mean_err`:  average uncertainty in magnitudes (namely typical survey uncertainty)

$$\bar{\epsilon} = \frac{1}{N} \sum_{i=1}^{N} \epsilon_i$$


- `median`: Median of magnitudes.

$$
\text{median}(m) = \begin{cases}
m_{\left(\frac{n+1}{2}\right)} & \text{if } n \text{ is odd} \\
\frac{m_{\left(\frac{n}{2}\right)} + m_{\left(\frac{n}{2}+1\right)}}{2} & \text{if } n \text{ is even}
\end{cases}
$$

Uses [`np.median`](https://numpy.org/doc/stable/reference/generated/numpy.nanmedian.html#numpy.nanmedian)
- `min`: Minimum value of the magnitudes, $m_{min}$.
- `max`: Maximum value of the magnitudes, $m_{max}$.
- `range`: range peak-to-peak amplitude of the magnitude values. Here, this is simply defined as the difference between the max and min (range) of magnitudes, $m_{max}-m_{min}$ . (for more robust definitions of amplitude, see the `VariabilityIndex` class below)
- `time_max`: Maximum value of the observation times ($t_{max}$).
- `time_min`: Minimum value of the observation times ($t_{min}$).
- `time_span`: Total time-span of the light curve, $t_{max}-t_{min}$ 
<!--
- `range`: another flavor of ptp amplitude bin in terms of maximum/minimum values of magnitude: $$x_{max}-x_{min}$$ 
-->


TODO:

- add further time info which can help the GLS definition. typical delta t, minimum/maximum delta t

## `FoldedLightCurve` class

Given that a real variable process, $\{m(t_i)\}$, is present in the observed light curve, we can assume that this process has an underlying characteristic timescale, $\tau$. Whether $\tau$ is a _periodic_ or an _aperiodic_ is the subject of the `TimeSeries` class. Here, whatever is the nature of $\tau$, we assume it can be used to transform the light-curve into a phase-folded representation,  $m_{\phi_i}=\{m(\phi_i)\}$, with $0\leq\phi_i\leq1$. In practice, the transformation $t_i\rightarrow \phi_i$ maps each time observation, $t_i$, into which _phase_ of the of a variability cycle that observation is. In this context, c is:
 

$$
    \phi_i=\frac{t_i-t_0}{\tau}-\Big\lfloor\frac{t_i-t_0}{\tau}\Big\rfloor,
$$

where $t_0$ is a reference epoch, and $\tilde{\phi_i}=\lfloor\frac{t_i-t_0}{\tau}\rfloor$ reflects the integer (floor) part of the ratio $\frac{t_i-t_0}{\tau}$. 

`FoldedLightCurve` will carry out this mapping:



```python
  from variability.lightcurve import  FoldedLightCurve
  lc_f = FoldedLightCurve(lc=lc, timescale=timescale)
```

where `timescale` is a timescale to be used for phase-folding the light-curve (for example, the variability period) and `lc` is a pre-defined `LightCurve` object. Alternatively, `FoldedLightCurve` can be created instead of a `LightCurve` object:

```python
  lc_f = FoldedLightCurve(time=time, mag=mag, err=err, timescale=period)
```

Note that creating `FoldedLightCurve` objects from `LightCurve` objects will result in shared memory location for (`lc.time`, `lc.mag`, `lc.err`) and (`lc_f.time`, `lc_f.mag`, `lc_f.err`).


 Additionally to the same attributes as a `LightCurve` object, `FoldedLightCurve` has the following additional attributes:

- `phase`: ($\{\phi_i\}$) phase values of the folded light curve ($0\leq\phi_i\leq1$).
- `phase_number`: ($\{\tilde{\phi_i}\}$) phase number is the integer part of the phase calculation. 
- `mag_pahsed`: ($m_{\phi_i}=\{m(\phi_i)\}$) magnitude values of the folded light curve, sorted based on ascending phase.
- `err_pahsed`: ($\epsilon_{\phi_i}=\{\epsilon(\phi_i)\}$) The error values of the folded light curve, sorted based on phase.
- `waveform`: ($\hat{m}_{\phi_i}$) waveform for phase-folded light curve estimated by smoothing the light curve using uneven_savgol (default) or other method if specified. 



- `residual`: residual curve obtained by subtracting the waveform from the phase-folded light curve:
  
  $$r_{\phi_i}=m_{\phi_i}-\hat{m}_{\phi_i}$$


**Note:** `FoldedLightCurve` requires a `timescale`, and uses as default the `uneven_savgol` waveform estimator, which requires parameters for window size (`window` set to 25\% the number of epochs as default) and the order of the polynomial fit used by the SavGol method (`polyorder=2`). Additionally, unless specified, `reference_time` is set to 0 when phase folding the light curve. Al these parameters can be updated on the go by the `lc_f.refold(timescale=None, reference_time=None, waveform_type=None, waveform_params=None)` method. For example, below, the `FoldedLightCurve` has its waveform estimator updated by keeping the `unevel_savgol` method, but using a polynomial of order 1 and a window with 10 epochs.

```python
lc_f.refold(waveform_type='uneven_savgol', waveform_params={'polynom':1, 'window':10})
```

 `FoldedLightCurve` objects inherit all the described properties from a `LightCurve` object, but has additional properties of its own:
- `timescale`: ($\tau$) The timescale used for folding the light curve. Can be a variability period or any characteristic timescale inferred for the light-curve (This can be inferred using the `timescale` module)
- `reference_time`: ($t_0$) Reference time for phase-folding the light-curve. It is set to 0 as default.
- `waveform_type`: waveform estimator used, `uneven_savgol` as default.
- `saunders_norm` Saunders metrics [(Saunders et al. 2006)](https://www.google.com/search?q=saunders+metrics+for+agent+observations+nasa+ads&num=10&sca_esv=7c40ceab512ed81b&sxsrf=AE3TifP_S5nlea_YZ3nVUFwLZSJpOuzfVA%3A1759502542255&ei=zuDfaLeoD5WA9u8P6JCG6Ak&ved=0ahUKEwi3x4zgoYiQAxUVgP0HHWiIAZ0Q4dUDCBA&uact=5&oq=saunders+metrics+for+agent+observations+nasa+ads&gs_lp=Egxnd3Mtd2l6LXNlcnAiMHNhdW5kZXJzIG1ldHJpY3MgZm9yIGFnZW50IG9ic2VydmF0aW9ucyBuYXNhIGFkczIFECEYoAFIpA9Q1AJYtA5wAXgAkAEAmAHlAaABkgiqAQU0LjQuMbgBA8gBAPgBAZgCCaACtAiYAwCIBgGSBwU0LjQuMaAHuB2yBwU0LjQuMbgHtAjCBwUxLjcuMcgHEA&sclient=gws-wiz-serp) diagnosis how _clumpy_ the phase coverage is compared to an ideal case of equally spaced sampling in phase. Based on a the size of the steps in phase between consecutive measurements, this is: 

$\Delta \phi_i =
\begin{cases}
\phi_{i+1} - \phi_i, & i = 1, \dots, N-1, \\[6pt]
(\phi_1 + 1) - \phi_N, & i = N;
\end{cases}
$

Then the metrics defined as:
$$
S_{\mathrm{norm}} \;=\; 
\frac{N+1}{\,N-1\,} \left( N \sum_{i=1}^{N} \left( \Delta \phi_i \right)^{2} - 1 \right).
$$

Where $S_{\mathrm{norm}} = 0$ for perfectly uniform phase coverage, $S_{\mathrm{norm}} \approx 1$ for random uniform phases, and $S_{\mathrm{norm}} => 1$ for clumpy or poor phase coverage.



**Note**: All returned values are sorted as a function of phase value. 

# Variability Indexes

Include a suite of widely used variability indexes. A great review on this subject is providedby  [Sokolovsky et al. (2017)](https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw2262) with several of the indexes implemented here discussed there with appropriate referencing. Here, `VariabilityIndex` concentrates on magnitude-domain features. (for time-domain features, see `TimeScale`).

## `VariabilityIndex`

To instantiate a `VariabilityIndex` object:

```python
from variability.indexes import VariabilityIndex

var = VariabilityIndex(lc_p, timescale=period, min_epochs=5)
```

you are expected to pass in a `LightCurve` object, or a `FoldedLightCurve` object. **Note that** some variability indexes, like the Q-index itself, require either a `timescale` argument or a `FoldedLightCurve` instance (which already have an instance `timescale`). `VariabilityIndex` requires a policy for minimum number of epochs in the light curve for properties derived from statistics with the light curve to be calculated, default to `min_epochs=5`. 

The list of implemented variability indexes currently implemented can be accessed with:

``VariabilityIndex._list_properties()``

This list depends if `VariabilityIndex` was created from a `LightCurve`, in which case all regular light-curve variability indexes implemented are listed; or if it was created from a `FoldedLightCurve(..., timescale)`, which phase-folded the light curve for the provided timescale. In this case, additionally to the regular indexes, with the addition of phase folded indexes. 

### regular light-curve Variability indexes:

These are variability index derived directly from the light curve (time, mag err).


#### Weighted Average `weighted_average`: 

Weighted average of the magnitude values using an uncertainty weight, $w_i=\frac{1}{\epsilon_i^2}$:

$$\bar{m}_w = \frac{\sum_{i=1}^{N} w_i m_i}{\sum_{i=1}^{N} w_i}$$

Uses [`numpy.average`](https://numpy.org/doc/stable/reference/generated/numpy.average.html#numpy.average).

#### Bias-corrected standard deviation (`std`)

Standard deviation of the magnitudes [(uses bias corrected `numpy.std`)](https://numpy.org/doc/stable/reference/generated/numpy.std.html).

$$
\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (m_i - \bar{m})^2}
$$

For non-variable sources, $\sigma\approx\bar{\epsilon}$, given that uncertainties $\{\epsilon_i\}$ are realistic.

#### signal-to-noise ratio `SNR` 

Defined here as the standard deviation of the data divided by average uncertainty. 

$$\text{SNR}=\frac{\sigma}{\bar{\epsilon}}$$



#### Shapriro-Wilk test (`VariabilityIndex.shapiro_wilk`)

$$W = \frac{\left(\sum_{i=1}^{n} a_i m_{(i)}\right)^2}{\sum_{i=1}^{n} (m_i - \bar{m})^2}$$

Where $W$ is the Shapiro-Wilk statistic, $n$ is the number of observations, $m_i$ are the individual values of the dataset, $\bar{m}$ are the mean (average) of the dataset, and $m_{(i)}$ are the $i$-th order statistic in the sorted dataset. The coefficients $a_i$ are pre-calculated constants based on the sample size and are used in the Shapiro-Wilk test. This is calculated using [`scipy.stats.shapiro`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html).

This is actually an index to test if the distribution is Gaussian. But note that this is used here as a ranking feature, rather than a statistical test, hence no p-value is returned. 

Expected behavior:
- For Gaussian noise (no variability): $W\approx 1$.
- For symmetric variability, $W\lesssim1$. 
- For highly asymmetric variability $W\ll 1$.

**Note:** It is good as a complementary index for capture asymmetric variability behaviour. 

#### median absolute deviation (MAD): (`VariabilityIndex.mad`)

  $$\text{MAD} = \text{median} \left( \left| m_i - \text{median}(m) \right| \right)$$

Calculate with [`scipy.stats.median_abs_deviation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html).

Expected behavior:
- For Gaussian noise: $1.4826\times\text{MAD}\approx\bar{\epsilon}$, i.e., this reflects the noise level. 
- For symmetric variability: $MAD>\bar{\epsilon}$.
- For asymmetric variability: $\sigma>MAD\gtrsim\bar{\epsilon}$, this is because unlike the standard deviation, $\sigma$, MAD is insensitive to outliers. 

**Note:** Insensitive to outliers (but also to real asymmetric variability) It is good for capturing symmetric behaviour. 

**Reference:** [Sokolovsky et al. (2017), Sec. 2.3](https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw2262)

#### $\chi^2$ (`VariabilityIndex.chi_square`)
$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}$$

with $\chi^2$ as the chi-squared statistic, $O_i$ is the observed frequency for each category or bin, $E_i$ is the expected frequency for each category or bin, and $k$ are the total number of categories or bins. This is a statistical tests to evaluate if the source is compatible with being constant. This is calculated following the equation:

$$\chi^2 = \sum_{i=1}^{N} \frac{\left( m_i - \bar{m}_w \right)^2}{\epsilon_i^2}, \quad
\bar{m}_w = \frac{\sum_{i=1}^{N} \frac{m_i}{\epsilon_i^2}}{\sum_{i=1}^{N} \frac{1}{\epsilon_i^2}}$$

 $\chi^2$ is a statistical test, where in the current context the null-hyposis is that the source is not variable (light curve is dominated by noise, assumed to be gaussian).

Expected behavior depends on the number of epochs (see reduced-$\chi^2$):
- For Gaussian noise: $\chi^2\approx N -1$
- For symmetric variability: $\chi^2> N -1$
- For asymmetric variability:$\chi^2\gg N -1$

**Note:** for non-normalised $\chi^2$, the values will depend on the number of epochs (degrees of freedon). $\chi^2$ grows with variability amplitude, but the $\left( x_i - \bar{x}_w \right)^2$ term implies that it rewards large outliers/asymmetries. Note that the assumption of gaussian noise here implies that this should better be estimated in flux, which my package is not doing at the moment, thus $\chi^2$ must be interpreted with a piece of salt. 

**Reference:** [Sokolovsky et al. (2017), Sec. 2.1](https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw2262)

#### reduced-$\chi^2$ (`VariabilityIndex.reduced_chi_square`)

$\chi_\nu^2 = \frac{\chi^2}{\nu}$, where $\nu$ are the degrees of freedom, which in this case relates to the number of epochs, $\nu=N-1$

Expected behavior:
- For Gaussian noise: $\chi^2\approx 1$
- For symmetric variability: $\chi^2> N -1$
- For asymmetric variability: $\chi^2\gg N -1$

**Reference:** [Sokolovsky et al. (2017), Sec. 2.1](https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw2262)

#### Inter-quantitle  range (IRQ) (`VariabilityIndex.iqr`)

$$\text{IQR} = Q_3 - Q_1$$
Where $Q_1$ and $Q_3$ are the first and third quartile. Estimated from [`scipy.stats.iqr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.iqr.html#scipy.stats.iqr). It is related to the $\sigma$ and to the $\text{MAD}$, where $\text{IQR}\approx0.761\text{MAD}$


Expected behavior:
- For Gaussian noise: $\text{IQR}\approx1.349\times\bar{\epsilon}$, i.e., this reflects the noise level. 
- For symmetric variability: $IQR>\bar{\epsilon}$.
- For asymmetric variability: $IQR\gtrsim\bar{\epsilon}$.

**Note:** Similar to the MAD, it captures variability without being sensitive to outliers, it may work better for asymmetric variability.

**Reference:** [Sokolovsky et al. (2017), Sec. 2.4](https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw2262)

#### Robust-Median Statistics (RoMS) (`VariabilityIndex.roms`)

$$\text{RoMS} = \frac{1}{N-1}\sum_{i=1}^{N} \frac{\big| m_i - \mathrm{median}(m)\big|}{\epsilon_i},$$

This is similar to the MAD, but it takes into consideration the typical uncertainty in the light curve. 

Expected behavior:
- For Gaussian noise: $RoMS\approx1$
- For real variables: $RoMS>1$

**Note:** It is not robust against outliers, as if the outlier has its uncertainty underestimated, it gives a large contribution to the RoMS. 

**Reference:** [Sokolovsky et al. (2017), Sec. 2.5](https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw2262)

#### normalised Excess Variance (`VariabilityIndex.normalised_excess_variance`)


$$\sigma_{\text{NXS}}^2 = \frac{\sigma^2 - \langle \bar{\epsilon}^2 \rangle}{\langle m \rangle^2}$$

It is normalized by the typical uncertainty, *i.e.,* subtracts the mean photometric noise from the data standard deviation and compares this to the mean magnitude. 

Expected behavior:
- For Gaussian noise: $\sigma_{\text{NXS}}^2\approx 0$
- For symmetric variability: $\sigma_{\text{NXS}}^2>0$
- For asymmetric variability: $\sigma_{\text{NXS}}^2\gg0$

**Note:** If uncertainties are too large (or over estimated), than $\sigma_{\text{NXS}}^2<0$ is possible. 

**Reference:** [Sokolovsky et al. (2017), Sec. 2.6](https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw2262)

#### Lag1AutoCorr ($l_1$) (`VariabilityIndex.lag1_auto_corr`)


$$l_1 = \frac{\sum_{i=1}^{N-1} (m_i - \bar{m})(m_{i+1} - \bar{m})}{\sum_{i=1}^{N} (m_i - \bar{m})^2}$$

First order autocorrelation coefficient. Measures how much a light curve is correlated with itself at one-time-step lag. 


Expected behavior:
- For Gaussian noise: consecutive points should be uncorrelated and $l_1\approx0$
- For slowly varying sources like periodic variables, there is small change between consecutive epochs and $l_1>0$.
- for rapid varying sources, $l_1$ can be anything really. 
- for monotonic variability, like a long-trend variable slowly increasing/decreasing magnitude, points probably do not change much in consecutive epochs and $l_1\rightarrow1$.

**Note:** This will depend on how the cadence of observations is.


#### Anderson-Darling (`VariabilityIndex.anderson_darling`)


$$A^2 = -N - \frac{1}{N} \sum_{i=1}^{N} \left[
\frac{2i - 1}{N} \ln F(m_{(i)}) +
\left( 1 - \frac{2i - 1}{N} \right) \ln \left( 1 - F(m_{(N-i+1)}) \right)
\right]$$

Where $A^2$ is the Anderson-Darling statistics, $n$ is the number of observations, $x_{(i)}$ is the $i$-th order statistic in the sorted dataset and $F(x_{(i)})$  is the cumulative distribution function at $x_{(i)}$, assuming a normal distribution. Estimated from [`scipy.stats.anderson`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html)


Expected behavior:
- For Gaussian noise: $A^2\approx0$ implies normally/gaussian distributed around the mean. 
- For symmetric variability: $A^2>0$, data is no longer normally distributed.
- For asymmetric variability: $A^2\gg0$, expect stronger non-normality. 

**Note**: If the approximation of Gaussian noise fails for the dataset, then even constant stars will show $A^2>0$. It is very sensitive to outliers. 


#### Skewness (`VariabilityIndex.skewness`)
$$\text{Skewness} = \frac{\frac{1}{n} \sum_{i=1}^{n} (m_i - \bar{m})^3}{\left(\frac{1}{n} \sum_{i=1}^{n} (m_i - \bar{m})^2\right)^{\frac{3}{2}}}$$

Expected behavior:
- For Gaussian noise: 
- For symmetric variability: 
- For asymmetric variability:

**Note**


####  kurtosis (`VariabilityIndex.kurtosis`)
$$\text{Kurtosis} = \frac{\frac{1}{n} \sum_{i=1}^{n} (m_i - \bar{m})^4}{\left(\frac{1}{n} \sum_{i=1}^{n} (m_i - \bar{m})^2\right)^2}$$


Expected behavior:
- For Gaussian noise: 
- For symmetric variability: 
- For asymmetric variability:

**Note**


#### Normalized peak-to-peak variability `VariabilityIndex.norm_ptp`

$$\nu = \frac{(m_i-\epsilon_i)_\mathrm{max} - (m_i-\epsilon_i)_\mathrm{min}}{(m_i+\epsilon_i)_\mathrm{max} + (m_i+\epsilon_i)_\mathrm{min}}$$

where $m_i$ is the magnitude measurement and $\epsilon_i$ is the corresponding measurement error. 


Expected behavior: measure of variability amplitude

**Reference:** [Sokolovsky et al. (2017), Sec. 2.7](https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw2262)



#### Tail peak to peak

Robust peak-to-peak amplitude estimator defined as the difference between the median values of the tails of the magnitude distribution. Tails are defined as the $p\%$ outermost sources at each side of the distribution.

$$\Delta m_{p}=\text{median}\{m_i:m_i\leq P_{100-p}\}-\text{median}\{m_i:m_i\leq P_{p}\},$$

Where $P_p$ is the p-th percentile of the distribution of magnitudes. 

Here there are 3 flavours implemented:

- `VariabilityIndex.ptp_5`: ptp based on the 5th and 95th tails
- `VariabilityIndex.ptp_10`: ptp based on the 10th and 90th tails
- `VariabilityIndex.ptp_20`: ptp based on the 20th and 80th tails

Expected behavior: measure of variability amplitude while robust against outliers. The smaller the percentile, the more sensitive it will be against extreme asymmetric variability. The higher p may work better for describing the typical variability amplitude. 

#### weighted standard deviation

**Reference:** [Sokolovsky et al. (2017), Sec. 2.2](https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw2262)

#### M-index (`VariabilityIndex.asymmetry_index`)

$$M = \frac{<m_{10\%}>-\text{median\{m\}}}{\sigma_m}$$

$<m_{10\%}>$ is all the data in the top and bottom decile of the light-curve. 
Not that there are conflicting definitions in the literature, where $\sigma_m$ is sometimes the overall rms of the light-curve and sometimes its standard-deviation! Here I am using the second one. 


Expected behavior:
- For symmetric variability: $M\sim0$
- For dimming variability: $M\gg0$
- For brightening variability: $M\ll0$


**Reference:** [Cody+2014](https://iopscience.iop.org/article/10.1088/0004-6256/147/4/82)

<!--
### Other variability indexes

#### Abbe

~~Calculate Abbe value as in Mowlavi 2014A%26A...568A..78M
https://www.aanda.org/articles/aa/full_html/2014/08/aa22648-13/aa22648-13.html~~

**Reference:** [Sokolovsky et al. (2017), also Sec. 2.16](https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw2262)

Also nami
-->

### Phase-folded variability indexes

These are indexes that required prior assumption on the timescale of variability. 


#### Q-index `VariabilityIndex.periodicity_index`

$$Q = \frac{\sigma_\mathrm{r}^2-\bar{\epsilon}^2}{\sigma^2-\bar{\epsilon}^2}$$ 

where:
- $\sigma^2$ and $\sigma^2_r$ are the ~~rms~~ variance values of the raw light curve and the phase-subtracted light curve.
- $\sigma^2$ is the variance of the original light-curve
- $\bar{\epsilon}$ is the mean photometric error


1. Find a period (Lomb Scargle Periodogram for ex)
2. Fold the light curve to the period
3. Use mooving average to get the smoothed shape of the curve
4. subtract it from phased light curve 
5. estimnate $\sigma_\mathrm{res}$


Expected behavior:
- For strictly periodic sources $Q\sim0$
- For aperiodic sources $Q\sim1$

**Reference:** [Cody+2014](https://iopscience.iop.org/article/10.1088/0004-6256/147/4/82)

**Note**

#### Means of residuals


## TO DO list

<details>
- :white_large_square: Fix Stetson-index implementation
- :white_large_square: Complete Variability-index documentation
  - :white_large_square: Complete description
  - :white_large_square: add examples
  - :white_large_square: add references
- interpercentile range
-  
</details>


# Filtering

Include a suite of filters to aid pre-processing light-curves. Most importantly, the `WaveForm` class provides a waveform estimation given the assumption of a periodic timescale of variability. 

**@juliaroquette** mostly implemented, but still needs polishing and debugging. 


## `Filtering`

```python
from variability.filtering import Filtering
```
<details>

### `Filtering.filter`

### `Filtering.sigma_clip`

### `Filtering.Cody_long_trend`

### `Filtering.savgol_longtrend`

### `Filtering.uneven_savgol`

### `Filtering.smooth_per_timescale`

### `Filtering.rolling_average`

</details>

## `WaveForm`

The class `WaveForm` provide tools to compute a smoothed version of a given phase-folded light-curve (`FoldedLightCurve`) based on a choice of filtering algorithm. This smoothed version is called 'waveform`.  Additionally, a residual light-curve is estimated as the phase-folded light-curve subtracted by its waveform. 


```python
from variability.filtering import  WaveForm
```

In the context of the variability indexes implemented here, `WaveForm` was designed with phase-folded light curves in mind. There is thus an underlying assumption that `phase` (phases for magnitudes in a phase-folded light curve) and `mag_phased` (phase-folded magnitudes) are circular. This assumption is used to deal with edging issues in most waveform-estimators implemented here.

To instantiate a `WaveForm` object:

```python
wf = WaveForm(phase, mag_phased)
```
A waveform can be obtained as:

```python
waveform = wf.get_waveform(waveform_type='uneven_savgol',
                          waveform_params={'window': 21, 'polynom': 2})
```

Where `waveform_type` is the waveform estimator of choice as in the list below, and `waveform_params` is a dictionary with the appropriate parameters for the chosen waveform estimator.


<details>

### `WaveForm.circular_rolling_average_number`

Simple rolling average that uses a fixed number of points for its window. Window size is defined by the attribute `window_size`, which should be an odd integer number. As default `window_size=5` but this should be chosen based on the light-curves at hand.

*Note:* This only works well with regularly phase sampling.

### `WaveForm.circular_rolling_average_phase`

Rolling average adapted to uneven based on a phase window. Useful for very irregular sampling in phase space. Window size is defined by the attribute `wd_phase` and represents an interval in phase. Default is `wd_phase=0.1`, i.e., 10% of the phase is used for windowing. 


### `WaveForm.waveform_H22`

	Boxcar smoothing method as used for waveform estimation in [Hillenbrand et al. 2022](https://ui.adsabs.harvard.edu/abs/2022AJ....163..263H/abstract). Borrowed on [the code used by Hillenbrand et al.](https://github.com/HarritonResearchLab/NAPYSOs) for waveform estimation in ZTF light curves. Uses [`astropy.convolution.Box1DKernel`](https://docs.astropy.org/en/stable/api/astropy.convolution.Box1DKernel.html). The only attribute here is `kernel=4` which is the same as using 1/4 of the number of points as kernel. 

### `WaveForm.waveform_Cody`

This uses Scipy's  median filter,
[`scipy.ndimage.median_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html), as used by [Cody et al.](https://ui.adsabs.harvard.edu/abs/2018AJ....156...71C/abstract). The attribute `n_point=50` is passed to `scipy.ndimage.median_filter` as the `size` attribute, which is just the window size for filtering. This is expected to work for evenly spaced data. 

### `WaveForm.savgol`

[Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) for uniformly evenly spaced curves. The Sav-Gol filter smooth the data by fitting polynomials of a given degree to adjacent point within a given windew. Here this simply employs [`scipy.signal.savgol_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) with all default parameters except for `window` (default 10) and `polynom` (=3) which . 

### `WaveForm.uneven_savgol`

Adaptation of the [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) for uneven data. This was based on an [implementation discussed at StackExchange](https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data). Similarly to the standard Savitzky-Golay filter, here the attributes `window` (default 10) and `polynom` (=3) define the window size and polynomial degree for filtering.


</details>


Once a waveform has been derived, a residual light curve can be estimated as:

```python
residuals = wf.residual_magnitude(waveform)
```
# Variability timescale 


## `timescale` module:

```python
from variability.timescale import  TimeScale
```

The `TimeScale` class allows to quick estimation of variability light curves for an object `LightCurve` defined from a trio (`time, mag, err`). There are two types of timescale estimator currently implemented:

- Periodic Timescale:
- Aperiodic timescale: 

## Structure functions

The **Structure Functions (SF)** are used to examine the timescales of variability in a light curve. It quantifies how the magnitude differences between observations evolve as a function of time lag.

#### **Definition**  

There are a few different definitions of SF out there, but here we work with definition as in [Hughes92](https://ui.adsabs.harvard.edu/abs/1992ApJ...396..469H) where:  

$$SF(\tau) = \frac{1}{N} \sum_{i,j} (m_i - m_j)^2, \,\, i>j$$

where $N$ is the number of pairs separated by a time lag $\tau$, and $m_i - m_j$ is the magnitude difference for each pair.


### **Key SF Properties**  

For the moment,  the SF analysis implemented is restricted to light curves with an underlying assumption of aperiodic variability (i.e., a Lomb-Scargle periodogram analysis is carried out in a previous step and no significant period was found).

Starting from this _aperiodic timescale_ hypothesis, we assume the structure function will have the form (`structure_function.model_function(x, t0, c0, c1)`):

$$y_{SF}(\tau) = C_0 + C_1(1 - \exp{(-\tau/t_0)}),$$


- At _small time lags_ ($\tau\rightarrow0$), the SF is dominated by measurement noise, which in this model is mapped to $C_0$
- At _intermediate time lags_, the SF increases following a linear slope in log-scale
- At _large time lags_ ($\tau\rightarrow t_{max}$), the SF saturates to a plateau ($C_0+C_1$). 
- The transition between the SF increase and the plateau occurs at the turnover timescale, $t_0$, which here we interporet as the characteristic timescale in the light curve. 

**Warnings:**

- This model assumes aperiodic variability. Periodic variability will introduce significant substructure to the SF and is not well represented by this model (see, for example [Roelens17](https://ui.adsabs.harvard.edu/abs/2017MNRAS.472.3230R) Figure 1.). 
- This model assumes there exist a dominant variability mode in the light-curve with a a timescale $t_{min}\leq t_s \leq t_{max}$. This model does not well represents multi-mode variability.

### SF sampling

Before fitting the SF model to the SF data. There may be a few complications behind sampling the data, which will depend of the specific light curve sampling. 


#### unbinned

The first step for sampling is to estimate the ${SF(\tau),\tau}$. This step is done by `structure_function.get_sf_time_lags(time, mag, err)`, and includes as a default the propagation of magnitude uncertainties into SF uncertainties.  

*Why not simply fit the SF model to unbinned data?* Depending on the duration and cadence of the light-curve, the SF has many have many more points at longer time-lags. Consequently, when fitting the the SF model directly to the unbinned data, $SF(\tau)$ for large $\tau$ will have much more weight in the fit.

#### Binned

The alternative to unbinned data is _to bin_ the data while averaging the SF values within bins. Binning will thus smooth regions of the SF where the density of points is too large compared to not so well populated $\tau$-ranges. In many applications "binning is sinning". But in this case sinning allows avoiding biasing the SF fits towards larger timescales. 

The function `adaptative_binning(time_lag, sf_values, sf_err=None)` includes different _flavours_ of binning. 

All _flavours_ of binning share an euristic approach for picking the number of bins to use based on a desired timescale resolution. This is determined by the `resolution` keyword. 
- In log scale (`log=True`) `resolution` ($\Delta_\tau$) should be in dex. The number of bins is then difined as:

$$N = 1 + \left\lfloor \frac{\log_{10}(\text{max}(\tau))-\log_{10}(\text{min}(\tau))}{\Delta_\tau} \right\rfloor$$

- in linear scale (`log=False`) `resolution` ($\Delta_\tau$) should be in the same units as $\tau$ (days)
 
$$N = 1 + \left\lfloor \frac{\text{max}(\tau)-\text{min}(\tau)}{\Delta_\tau} \right\rfloor$$


This calculation is done within the auxiliary function `bin_edges_and_centers(time_lag, resolution, log=True)`, which supports the binning processing by properly deriving the number of bins and making sure the edge values of $\tau$ are the center of the last bins. 

**what resolution to use**?

For example, relatively well sampled SF around timescale related to rotational modulation in young stars, we may want to have timebins 0.5 days appart at a $\tau=10$ d

$$\log_{10}{(10 + 0.5)} - \log_{10}{(10)} \approx 0.02\,\text{dex}$$

 which results on $N_{bins}=240$

##### regular bins

Simple regular bins will be problematic for most light curves because some SF bins will be empty, and others will not have enough data to average. Still for comparison and completeness, the `adaptative_binning` function will yield regular bins if `adaptative_binning(time_lag, sf_values, hybrid=False, bin_min_size=1, max_bin_exp_factor=1.0`)

#####  Adaptative Binning

Bins are adapted according to some criteria.

`structure_function.adaptative_binning(time_lag, sf_values, sf_err=None,
                       log=True,
                       hybrid=False,
                       bin_min_size=5, 
                       max_bin_exp_factor=3.0, 
                       step_size=0.2, 
                       resolution=0.02)`

###### Expanded bins 

Bins are expanded around its centre until either it is composed by minimum number of data-points (`bin_min_size`) or it is completely merged with its two imediate neighbours.
This is similar to the approach in [Roelens17](https://ui.adsabs.harvard.edu/abs/2017MNRAS.472.3230R) where a tolerance factor $\epsilon_i$ introduced to ensure that each bin contains enough pairs for reliable calculation. 

The maximum expansion allowed is set by `max_bin_exp_factor` which should be a value between `1.0` and `3.0` where:

- `max_bin_exp_factor=1` allows for no expansion (bins are fixed), 
- `max_bin_exp_factor=3.0` means 3 bins are merged (the one in question and each of its imediate bins)
- `max_bin_exp_factor=2.0` means the bin is expanded until half of their neighour bins.

Additionally, `step_size` sets the size of the expansion step, where:
- `step_size=0.05` is the minimum value allowed and sets a 2.5% expansion to each direction at each iteration
- `step_size=1.0` is the maximum and it means the bin grows 50% to each direction. 

If `bin_min_size` is never reached, the bin is discarded. In this context, `pair_counts` return the number of sources per bin, while `irregular_bin=0.0` if this was a regular bin, `irregular_bin=-1.0` if using individual pairs and `>0.0` it reflects final width of the bin after expansion. 

For example: `adaptative_binning(time_lag, sf_values, sf_err=None, hybrid=False, bin_min_size=5, max_bin_exp_factor=3.0, step_size=0.2)` will expand bins until either the bin is merged with its immediate neighbours or at least `bin_min_size=5` are within the bin, while gradually expanding bins by 20% of their width. 

###### Hybrid mode

For bins where the number of sources is lower than `bin_min_size`, the data points in that bin are appended rather than their average. `adaptative_binning(time_lag, sf_values, sf_err=None, hybrid=True, bin_min_size=5)` will count indivual points in bins with less than ` bin_min_size=5` sources. (see note on treatment of uncertainty)


##### Log/linear scale:

Binning can be done in either log (`log=True`) or linear scale (`log=False`). However note that the linear case won't work well for SF where $\tau$ covers too many orders of timescale, such as the use of linear binning will squeeze all small $\tau$ into a few bins. 

##### treatment of uncertainty

- When `sf_err=sf_error` uncertainty is propagated (from the photometric uncertainties) from the SF into the bins as  $\Delta \text{SF}_{bin} = \frac{\sqrt{\sum \sigma_{SF}^2}}{N}$. However, this will often underestimate uncertainties and bias SF-fits.

- when `sf_err=None` uncertainties will be ignored and the bin uncertainty will be estimated from the spread in the bin as $\Delta \text{SF}_{bin} = \frac{\sigma_{SF_{bin}}}{\sqrt{N}}$.
  - When hybrid mode is also used (`sf_err=None, hybrid=True`), the median absolute deviation of the whole set of `sf_values` is used rather than the SF propagate uncertainties: $\Delta \text{SF}_{ind} = 1.4826\times\text{median}(|SF-\tilde{SF}|)$. 

### SF-Fitting 

The fitting function, $ y_{SF}(\tau)$, was defined above. For fitting it to the SF data, we are using the MINUIT optimisation tool from CERN. 

- [MINUIT reference manual](https://root.cern.ch/download/minuit.pdf)  
- [iminuit library](https://scikit-hep.org/iminuit/)

Minuit is just a minimisation function and will use whatever a function (`cost_function_for_minuit`) I pass `m=Minuit(cost_function_for_minuit)`. Here I simply adapted Chloé's `fit_with_minuit` to take the refactored `cost_function`. 


#### Cost-functions for minimization:

- $y(\tau)$ is the SF data, with $t_i$ the SF data-point at the i-th $\tau$.
- $\sigma_i$ is the uncertainty/error at the i-th SF data point.
- $\sigma_y$ is the standard deviation for the full SF dataset
- $m\rightarrow y_\text{SF}$ is the model assumed for the SF


##### L2 (simple residual)

This is a simple for minimizing residuals. Can work well enough for minimization, but it does not take noise into consideration and it is is not really a statistical likehood function. It will work well only as long as the errors are regular/well-behaved (gausian)

$$\chi^2_\mathrm{L2} = \sum_{i} \bigl(y_i - m_i\bigr)^2.$$ 

$$\chi^2_\mathrm{L2,log} = \sum_{i} \bigl(\log_{10}(y_i) - \log_{10}(m_i)\bigr)^2.$$ 

##### L2 weighted by error ($\chi^2$):

This is the usual $\chi^2$ definition. Though it works well if errors are gaussian. If uncertainties are weird, which seems to be the case here, may not be the best.

$$\chi^2_\mathrm{L2,error} = \sum_{i} \left( \frac{y_i - m_i}{\sigma_i} \right)^2$$
  

##### L2 with error in log-scale

In log-scale, $y\rightarrow\log_{10}(y)$, but error needs to be properly propagated. Note that log-scale overweights the small lags (uncertainty propagation gets super weird for small y).


$$\sigma_{\log y,i} = \frac{\sigma_i}{y_i \ln 10}$$

$$
\chi^2_\mathrm{L2, error,log} = \sum_{i} \left( \frac{\log_{10}(y_i) - \log_{10}(m_i)}{\sigma_{\log y,i}} \right)^2.$$

##### Reduced $\chi^2$

For reduced-\Chi^2:

$$\chi_{red}^2 = \frac{\chi^2}{N-k}$$

Where $N$ is the number of datapoints in $y$ and $k$ is the number of fitting parameters in the model. In our case here $k=3$ since we have the parameters $(C_0, C_1, t_0)$. 

This reduced form allows to at least evaluate how bad the errors are, as we should have:

- $\chi_{red}^2\approx 1$ for good fits
- $\chi_{red}^2\gg 1$ for underfit with error likely understimated
- $\chi_{red}^2\ll 1$ for overfit with overestimated errors 

  
##### L1 norm:

The L1 norm offers an alternative where the residuals are not squared such as the cost-function is less penalised by outliers. 

$$\text{L1}= \sum_{i} \bigl|y_i - m_i\bigr|$$ 



##### L1 norm w/ error 

Similarly to the L2 version, we can weight it by uncertainties. Noting that similarly to the L2 version this will work well as long as the uncertainties are good estimates of the real uncertainty. 

$$
\text{L1}_{\text{error}} = \sum_{i=1}^{N} \frac{\lvert y_i - m_i \rvert}{\sigma_i}
$$


##### L1 norm log-scale w/error:

To deal with the several orders of magnitude covered by both SF and $\tau$, logscale is a good option, though it may overweight small lags

$$
\text{L1}_{\text{log,weighted}} = \sum_{i=1}^{N} \frac{\lvert \log_{10}(y_i) - \log_{10}(m_i) \rvert}{\dfrac{\sigma_i}{y_i \ln(10)}}
$$


## TO DO list
- :heavy_check_mark: **@juliaroquette** Polish Lomb Scargle 
- :white_large_square: **@juliaroquette** Add Structure function 
- :white_large_square: Include alternative fitting to iminuit?
<!-- - :white_large_square: **@juliaroquette**  -->
<!-- - :white_large_square: **@juliaroquette**  -->


**@juliaroquette** Still under implementation, will include our codes for estimating timescales. 


## `SyntheticLightCurve`
<details>
**@juliaroquette** Still under implementation, will allow to generate synthetic light-curves for given observational windows. 

```python 
from variability.lightcurve import FoldedLightCurve
```

## TO DO list

:white_large_square: **@juliaroquette** It may be worth it consider the possibility of merging `LightCurve` and `FoldedLightCurve` into a single class. <- Consider that after the `timescale.py` package has been implemented. 

:white_large_square: read observational windows from file

:white_large_square: Implement a list of observational windows
  - :white_large_square: TESS
  - :white_large_square: Rubin
  - :heavy_check_mark: ZTF
  - :heavy_check_mark: ASAS-SN (g and V)
  - :heavy_check_mark: Gaia-DR3
  - :heavy_check_mark: Gaia-DR4
  - :heavy_check_mark: Gaia-DR5
  - :heavy_check_mark: AllWISE
  - :white_large_square: input

:white_large_square: Include waveforms 
  - :heavy_check_mark: PS
  - :white_large_square: QPS
  - :white_large_square: EB
  - :white_large_square: AATAU
  - :white_large_square: QPD
  - :white_large_square: QPB
  - :white_large_square: B
  - :white_large_square: MP
  - :white_large_square: LP

:white_large_square: Function that generates a waveform for a refined timestep

  - :white_large_square: Function that convolves the waveform to an observational window
</details>

