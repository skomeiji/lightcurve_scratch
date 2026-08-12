import numpy as np

plx = 0.5395
gmag = 12.829108

mg = gmag - 5 * np.log10(plx)

L = 10 ** ((4.84 - mg) / 2.5)

print(L)