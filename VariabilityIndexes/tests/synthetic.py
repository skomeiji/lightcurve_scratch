"""
Produces a synthetic dataset for testing purposes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


mag_sin = 0.5*amplitude*np.sin(2*np.pi*time/period) + np.random.random_sample(N)*.25


