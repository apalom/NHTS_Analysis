# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 09:28:58 2018

@author: Alex
"""

'''
import numpy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = dfRk1.iloc[:,0]

hist, bin_edges = numpy.histogram(data, density=True)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

#fit type


# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*numpy.exp(-(x-mu)**2/(2.*sigma**2))

# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [1., 0., 1.]

coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)

# Get the fitted curve
hist_fit = gauss(bin_centres, *coeff)

plt.plot(bin_centres, hist, label='Test data')
plt.plot(bin_centres, hist_fit, label='Fitted data')

# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
print('Fitted mean = ' % coeff[1])
print('Fitted standard deviation = ' % coeff[2])


plt.show()

'''

from scipy.stats import norm
from numpy import linspace
from pylab import plot,show,hist,figure,title

# picking 150 of from a normal distrubution
# with mean 0 and standard deviation 1
data = dfRk3.iloc[:,0].as_matrix()
samp = data

#samp = norm.rvs(loc=0,scale=1,size=150) 

mu, std = norm.fit(samp) # distribution fitting

# now, param[0] and param[1] are the mean and 
# the standard deviation of the fitted distribution
x = linspace(min(samp),max(samp),100)
# fitted distribution
pdf_fitted = norm.pdf(x,loc=mu,scale=std)
# original distribution
pdf = norm.pdf(x)

title('Normal distribution')
plot(x,pdf_fitted,'r-',x,pdf,'b-')
hist(samp,normed=1,alpha=.3)
show()

print('Norm.Dist Mean: %0.3f ' % mu)
print('Norm.Dist Standard Deviation: %0.3f ' % std)
