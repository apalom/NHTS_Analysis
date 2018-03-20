# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:18:38 2018

@author: Alex

Normal Distribution Fit of Data
"""

from scipy.stats import norm
from numpy import linspace
from pylab import plot,show,hist,figure,title

# picking 150 of from a normal distrubution
# with mean 0 and standard deviation 1
data = dfRk3.iloc[:,0].as_matrix()
samp = data

#samp = norm.rvs(loc=0,scale=1,size=150) 

mu, std = norm.fit(samp) # distribution fitting

mu1, std1, skew, kurt = norm.stats(loc = mu, scale = std)

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