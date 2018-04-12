# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:45:02 2018

@author: Alex Palomino
"""
# import libraries
import pandas as pd
import timeit
import matplotlib.pyplot as plt

timeExec= []
start_time = timeit.default_timer()

# Column names for filter
colNames = ['TDCASEID','STRTTIME','ENDTIME','WHYTRP1S']

# NHTS2009 Data Location for Alex's Lab Computer
df0 = pd.read_csv(r'C:\Users\Alex\Documents\NHTS_2017\trippub.CSV', header=0)

dfNHTS = df0.filter(colNames, axis = 1)
firstNHTSrows = dfNHTS.head(25) 



elapsed = timeit.default_timer() - start_time
timeExec.append(elapsed)

print('Execution time: {0:.2f} sec'.format(elapsed))