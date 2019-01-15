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

#%% Data Filters

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

#%% TRIPMILE HISTOGRAM

qE_high = data.TRPMILES.quantile(0.9545); #remove 2 std dev outlier
qE_low = data.TRPMILES.quantile(1-0.9545); #remove 2 std dev outlier

df_miles = data[(data.TRPMILES > qE_low) & (data.TRPMILES < qE_high)]

binEdges = np.arange(0, qE_high, 1)
numBins = int(np.sqrt(len(df_miles.TRPMILES)));

n, bins, patches = plt.hist(x=df_miles.TRPMILES, bins=binEdges, density=True, color='grey', rwidth=0.85)

plt.xlabel('Trip Miles (mi)')
plt.xticks(np.arange(0, qE_high, 2))
plt.ylabel('Frequency')
plt.title('NHTS 2017 Trip Miles')

#%% Charging Need

df_time1 = df_time.filter(['HOUSEID', 'VEHID', 'TDTRPNUM', 'TDCASEID', 'DWELTIME', 'TRPMILES'], axis = 1)

df_time1 = df_time1.reset_index(drop=True)

df_time1 = df_time1.sort_values(by=['HOUSEID', 'VEHID', 'TDTRPNUM'])

kwhPerMile = 0.33

houses = list(set(df_time1.HOUSEID))

df_chgrNeed = pd.DataFrame(np.zeros((len(df_time1),2)), columns=['TDCASEID', 'CHARGERKW'])

i = 0;

for h in houses:
    dfTemp = df_time1[df_time1.HOUSEID == h]
    
    if len(dfTemp) == 1:
        chgrNeed = 2 * dfTemp.TRPMILES * kwhPerMile * 60 / dfTemp.DWELTIME
        caseID = int(dfTemp.iloc[0].TDCASEID)
        
        print('a', i)        
        df_chgrNeed.iloc[i] = [caseID, chgrNeed]
        i += 1
    
    else: 
        vehicles = list(set(dfTemp.VEHID))
    
        for v in vehicles:
            dfTemp1 = dfTemp[dfTemp.VEHID == v] 
            
            if len(dfTemp1) == 1:
                chgrNeed = 2 * dfTemp1.iloc[0].TRPMILES * kwhPerMile * 60 / dfTemp1.iloc[0].DWELTIME
                caseID = int(dfTemp1.iloc[0].TDCASEID)
                
                print('b', i)
                df_chgrNeed.iloc[i] = [caseID, chgrNeed]
                i += 1
                
            else:
                for j in range(len(dfTemp1)):
                    if j < len(dfTemp1)-1:
                        dist = dfTemp1.iloc[j].TRPMILES + dfTemp1.iloc[j+1].TRPMILES
                        
                        chgrNeed = 2 * dist * kwhPerMile * 60 / dfTemp1.iloc[j].DWELTIME
                        caseID = int(dfTemp1.iloc[0].TDCASEID)
                        
                        print('c', i)
                        df_chgrNeed.iloc[i] = [caseID, chgrNeed]
                        i += 1
            
                    if j == len(dfTemp1)-1:
                        
                        chgrNeed = 2 * dfTemp1.iloc[j].TRPMILES * kwhPerMile * 60 / dfTemp1.iloc[j].DWELTIME
                        caseID = int(dfTemp1.iloc[j].TDCASEID)
                        
                        print('d', i)
                        df_chgrNeed.iloc[i] = [caseID, chgrNeed]
                        i += 1

#%% CHARGERKW Histogram
                        
qE_high = df_chgrNeed.CHARGERKW.quantile(0.9545); #remove 2 std dev outlier
qE_low = df_chgrNeed.CHARGERKW.quantile(1-0.9545); #remove 2 std dev outlier

df_need = df_chgrNeed[(df_chgrNeed.CHARGERKW > qE_low) & (df_chgrNeed.CHARGERKW < qE_high)]

binEdges = np.arange(0, qE_high, 5)
numBins = int(np.sqrt(len(df_need.CHARGERKW)));

n, bins, patches = plt.hist(x=df_need.CHARGERKW, bins=binEdges, density=False, color='grey', rwidth=0.85)

plt.xlabel('Charger Demand (kW)')
plt.xticks(np.arange(0, qE_high, 10))
plt.ylabel('Frequency')
plt.title('NHTS 2017 Trip Charger Need')


#%% Export data

df_need.to_csv('nhts2017_chgrNeed_removeOutlier.csv')



