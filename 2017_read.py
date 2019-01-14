# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 10:27:48 2019

@author: Alex
"""

import numpy as np
from scipy.stats import mstats
import matplotlib.pyplot as plt
import pandas as pd
from os import path
import timeit
import time
import datetime

#%% Import System Data

# Raw Data
filePath = r'C:\Users\Alex\Box Sync\Alex and Masood\WestSmartEV\Data from Partners\NHTS 2017\trippub.csv';
# Import Data
dataRaw = pd.read_csv(filePath);
data = dataRaw;
dataHead = data.head(100);
dataTypes = data.dtypes;

allColumns = list(data);

#%%


data = data[data.TRPMILES > 0]
data = data[(data.TRPTRANS > 0) & (data.TRPTRANS < 6)] #6 is RV transport
data = data[data.DWELTIME > 0]

#data1 = data.filter(['Start', 'End', 'Duration', 'Charging'], axis = 1)

#%% DWELTIME HISTOGRAM

qE_high = data.DWELTIME.quantile(0.9545); #remove 2 std dev outlier
qE_low = data.DWELTIME.quantile(1-0.9545); #remove 2 std dev outlier

df_time = data[(data.DWELTIME > qE_low) & (data.DWELTIME < qE_high)]

binEdges = np.arange(0, qE_high, 15)
numBins = int(np.sqrt(len(df_time.DWELTIME)));

n, bins, patches = plt.hist(x=df_time.DWELTIME, bins=binEdges, density=True, color='grey', rwidth=0.85)

plt.xlabel('Dweltime (min)')
plt.xticks(np.arange(0, qE_high, 60))
plt.ylabel('Frequency')
plt.title('NHTS 2017 Dwell Time')


#%% DWELTIME HISTOGRAM