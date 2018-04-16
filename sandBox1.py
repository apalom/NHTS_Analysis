# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 09:28:58 2018

@author: Alex
"""

import scipy
from scipy import stats
from scipy.stats import norm
#from pylab import plot,show,hist,figure,title
import matplotlib.pyplot as plt

x = matK3[:,0]
x.sort()

x = x*max0

param = norm.fit(x) # distribution fitting
# now, param[0] and param[1] are the mean and 
print('Mean:', param[0])
print('Std Dev:', param[1])

# fitted distribution
pdf_fitted = norm.pdf(x,loc=param[0],scale=param[1])

# original distribution
pdf = norm.pdf(x)

plt.title('Normal distribution')
plt.plot(x,pdf_fitted,'b-')
plt.hist(x,normed=1,alpha=.3)
plt.show()



