# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 11:33:32 2018

@author: Alex
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
'''
x = x1
num_bins = 10
counts, bins = np.histogram(x, bins=num_bins)
bins = bins[:-1] + (bins[1] - bins[0])/2
probs = counts/float(counts.sum())
print(probs.sum()) # 1.0
plt.bar(bins, probs, 1.0/num_bins)
plt.show()
'''

from scipy.stats import poisson
import numpy as np 
import matplotlib.pyplot as plt

x= x1
plt.plot(x, norm.pmf(x,502))

plt.show()
