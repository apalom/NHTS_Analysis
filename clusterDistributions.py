# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 09:28:58 2018

@author: Alex
"""

from scipy import stats
from scipy.stats import norm
#from pylab import plot,show,hist,figure,title
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


# Function to convert histogram plot to %
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(scale * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


max0 = dfPlot.max()[0]  

x1 = matK1[:,0]
x1.sort()
x1 = x1*max0*(1440/max0)

param1 = norm.fit(x1) # distribution fitting
# now, param[0] and param[1] are the mean and 
#print('Mean:', param[0])
#print('Std Dev:', param[1])

# fitted distribution
pdf_fitted1 = norm.pdf(x1,loc=param1[0],scale=param1[1])
# original distribution
pdf1 = norm.pdf(x1)

x2 = matK2[:,0]
x2.sort()
x2 = x2*max0*(1440/max0)
param2 = norm.fit(x2)
pdf_fitted2 = norm.pdf(x2,loc=param2[0],scale=param2[1])
pdf2 = norm.pdf(x2)

x3 = matK3[:,0]
x3.sort()
x3 = x3*max0*(1440/max0)
param3 = norm.fit(x3)
pdf_fitted3 = norm.pdf(x3,loc=param3[0],scale=param3[1])
pdf3 = norm.pdf(x3)

x4 = matK4[:,0]
x4.sort()
x4 = x4*max0*(1440/max0)
param4 = norm.fit(x4)
pdf_fitted4 = norm.pdf(x4,loc=param4[0],scale=param4[1])
pdf4 = norm.pdf(x4)

'''
x5 = matK5[:,0]
x5.sort()
x5 = x5*max0
param5 = norm.fit(x5)
pdf_fitted5 = norm.pdf(x5,loc=param5[0],scale=param5[1])
pdf5 = norm.pdf(x5)

x6 = matK6[:,0]
x6.sort()
x6 = x6*max0
param6 = norm.fit(x6)
pdf_fitted6 = norm.pdf(x6,loc=param6[0],scale=param6[1])
pdf6 = norm.pdf(x6)
'''


plt.figure(figsize=(8,6))

plt.plot(x1,pdf_fitted1,'b-', label='K1')
plt.hist(x1,normed=1,alpha=.3)
#bgrcmy

plt.plot(x2,pdf_fitted2,'g-', label='K2')
plt.hist(x2,normed=1,alpha=.3)

plt.plot(x3,pdf_fitted3,'r-', label='K3')
plt.hist(x3,normed=1,alpha=.3)

plt.plot(x4,pdf_fitted4,'c-', label='K4')
plt.hist(x4,normed=1,alpha=.3)
'''
plt.plot(x5,pdf_fitted5,'m-', label='K5')
plt.hist(x5,normed=1,alpha=.3)

plt.plot(x6,pdf_fitted6,'k-', label='K6')
plt.hist(x6,normed=1,alpha=.3)
'''


# Create the formatter using the function to_percent. This multiplies all the
# default labels by 100, making them all percentages
scale = 1000
formatter = FuncFormatter(to_percent)

# Set the formatter
#plt.yticks(np.arange(0, 0.00275, step=0.00025))
plt.title('Work k++ Dwell Time Normal Distributions')
plt.gca().yaxis.set_major_formatter(formatter)
plt.xlabel('ENDTIME (minutes)')
plt.xlim([0,1440])
plt.xticks(np.arange(0, 1440, step=120))
plt.legend(bbox_to_anchor=(0.92, 0.92)) 

plt.show()



