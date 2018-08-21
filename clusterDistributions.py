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
import statsmodels.api as sm
import statsmodels.formula.api as smf


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

# %% Distribution of Trip Miles
        
#max0 = dfPlot.max()[1]  
max0 = 30

x1 = matK1[:,1]
x1.sort()
x1 = x1*max0
#x1 = x1*max0*(1440/max0)

param1miles = norm.fit(x1) # distribution fitting
# now, param[0] and param[1] are the mean and 
#print('Mean:', param[0])
#print('Std Dev:', param[1])

# fitted distribution
pdf_fitted1 = norm.pdf(x1,loc=param1miles[0],scale=param1miles[1])
# original distribution
pdf1 = norm.pdf(x1)

x2 = matK2[:,1]
x2.sort()
x2 = x2*max0
#x2 = x2*max0*(1440/max0)
param2miles = norm.fit(x2)
pdf_fitted2 = norm.pdf(x2,loc=param2miles[0],scale=param2miles[1])
pdf2 = norm.pdf(x2)

x3 = matK3[:,1]
x3.sort()
x3 = x3*max0
#x3 = x3*max0*(1440/max0)
param3miles = norm.fit(x3)
pdf_fitted3 = norm.pdf(x3,loc=param3miles[0],scale=param3miles[1])
pdf3 = norm.pdf(x3)

x4 = matK4[:,0]
x4.sort()
x4 = x4*max0
#x4 = x4*max0*(1440/max0)
param4miles = norm.fit(x4)
pdf_fitted4 = norm.pdf(x4,loc=param4miles[0],scale=param4miles[1])
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
plt.hist(x1,bins=10,normed=1,alpha=.3, color='blue')
#bgrcmy

plt.plot(x2,pdf_fitted2,'g-', label='K2')
plt.hist(x2,bins=10,normed=1,alpha=.3, color='green')

plt.plot(x3,pdf_fitted3,'r-', label='K3')
plt.hist(x3,bins=10,normed=1,alpha=.3, color='crimson')

plt.plot(x4,pdf_fitted4,'c-', label='K4')
plt.hist(x4,bins=10,normed=1,alpha=.3, color='skyblue')

'''
plt.plot(x5,pdf_fitted5,'m-', label='K5')
plt.hist(x5,normed=1,alpha=.3)

plt.plot(x6,pdf_fitted6,'k-', label='K6')
plt.hist(x6,normed=1,alpha=.3)
'''


# Create the formatter using the function to_percent. This multiplies all the
# default labels by 100, making them all percentages
scale = 1
formatter = FuncFormatter(to_percent)

# Set the formatter
#plt.yticks(np.arange(0, 0.00275, step=0.00025))
plt.title('Other k++ Normal Distributions [Trip Miles]')
#plt.gca().yaxis.set_major_formatter(formatter)
plt.xlabel('Trip Distance (miles)')
plt.legend(bbox_to_anchor=(0.92, 0.92)) 

plt.show()

# %% Distribution of End Time 
        
max0 = dfPlot.max()[0]  

x1 = matK1[:,0]
x1.sort()
x1 = x1*max0*(1440/max0)

param1dwell = norm.fit(x1) # distribution fitting
# now, param[0] and param[1] are the mean and 
#print('Mean:', param[0])
#print('Std Dev:', param[1])

# fitted distribution
pdf_fitted1 = norm.pdf(x1,loc=param1dwell[0],scale=param1dwell[1])
# original distribution
pdf1 = norm.pdf(x1)

x2 = matK2[:,0]
x2.sort()
x2 = x2*max0*(1440/max0)
param2dwell = norm.fit(x2)
pdf_fitted2 = norm.pdf(x2,loc=param2dwell[0],scale=param2dwell[1])
pdf2 = norm.pdf(x2)

x3 = matK3[:,0]
x3.sort()
x3 = x3*max0*(1440/max0)
param3dwell = norm.fit(x3)
pdf_fitted3 = norm.pdf(x3,loc=param3dwell[0],scale=param3dwell[1])
pdf3 = norm.pdf(x3)

x4 = matK4[:,0]
x4.sort()
x4 = x4*max0*(1440/max0)
param4dwell = norm.fit(x4)
pdf_fitted4 = norm.pdf(x4,loc=param4dwell[0],scale=param4dwell[1])
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
plt.hist(x1,bins=10,normed=1,alpha=.3, color='blue')
#bgrcmy

plt.plot(x2,pdf_fitted2,'g-', label='K2')
plt.hist(x2,bins=10,normed=1,alpha=.3, color='green')

plt.plot(x3,pdf_fitted3,'r-', label='K3')
plt.hist(x3,bins=10,normed=1,alpha=.3, color='crimson')

plt.plot(x4,pdf_fitted4,'c-', label='K4')
plt.hist(x4,bins=10,normed=1,alpha=.3, color='skyblue')
'''
plt.plot(x5,pdf_fitted5,'m-', label='K5')
plt.hist(x5,normed=1,alpha=.3)

plt.plot(x6,pdf_fitted6,'k-', label='K6')
plt.hist(x6,normed=1,alpha=.3)
'''


# Create the formatter using the function to_percent. This multiplies all the
# default labels by 100, making them all percentages
scale = 100
formatter = FuncFormatter(to_percent)

# Set the formatter
#plt.yticks(np.arange(0, 0.00275, step=0.00025))
plt.title('Other k++ Trip Normal Distributions [End Time]')
#plt.gca().yaxis.set_major_formatter(formatter)
plt.xlabel('ENDTIME (minutes)')
plt.xlim([0,1440])
plt.xticks(np.arange(0, 1440, step=120))
plt.legend(bbox_to_anchor=(0.87, 0.92)) 

plt.show()

