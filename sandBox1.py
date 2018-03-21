# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 09:28:58 2018

@author: Alex
"""

from scipy import stats

x = np.linspace(-15, 15, 9)
stats.kstest(x, 'norm')
