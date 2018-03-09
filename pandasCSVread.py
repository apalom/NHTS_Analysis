# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 22:57:53 2018

@author: Alex Palomino
"""

# import libraries
import pandas as pd
import timeit
import matplotlib.pyplot as plt

# initialize values
start_time = timeit.default_timer()
whyID = {}
whyIDsum = {}
whyIDsumList = []

# NHTS2009 Data Location for Alex's Laptop 
df0 = pd.read_csv(r'C:\Users\avi_b\Box\CS6140 Project\Data\CSV\DAYV2PUB.CSV', header=0)

# NHTS2009 Data Location for Alex's Lab Computer
# df0 = pd.read_csv(r'C:\Users\Alex\Google Drive\Classes\18_Spring\CS6140 Data Mining\CS6140 Project\Data\CSV\DAYV2PUB.CSV', header=0)

# filter dataframe zero (raw NHTS2009) to columns listed in filter
df1 = df0.filter(['TDCASEID','TRAVDAY','STRTTIME','DWELTIME','ENDTIME','TRIPPURP',
                  'WHYFROM','WHYTO','WHYTRP1S','WHYTRP90','WHODROVE',
                  'CENSUS_D','CENSUS_R','DRIVER','AWAYHOME','FRSTHM','TDTRPNUM',
                  'TDWKND','TRPACCMP','TRPHHACC','TRVLCMIN','TRVL_MIN','TRWAITTM',
                  'VEHTYPE','VEHYEAR','VMT_MILE','HHFAMINC','HHSIZE','HHSTATE','HOMEOWN',
                  'NUMADLT','NUMONTRIP','PRMACT','PAYPROF','PROXY','PRMACT','R_AGE','R_SEX'], axis=1)

# function call to attribute why descriptions with why codes
from funcWhyID import funcWhyID
[df1, whyID, whyIDsum] = funcWhyID(df1, whyID, whyIDsum)
whyIDsumList = set(df1['whyDescSmry'])


# build out dataframe table
colNames0 = list(df0) # shows all column headers
colNames1 = list(df1) # shows all column headers
firstNrows0 = df0.head(25) # shows first n rows
firstNrows1 = df1.head(25) # shows first n rows
lastNrows0 = df0.tail(5) # shows last n rows
lastNrows1 = df1.tail(5) # shows last n rows

df0['TRIPPURP'].describe()

# print data shapes (rows x columns)
print('Dataframe Raw Shape:', df0.shape)
print('Dataframe Filtered Shape:', df1.shape)

elapsed = timeit.default_timer() - start_time

# timeit statement
print('Execution time: {0:.4f} sec'.format(elapsed))

# %% plotting section

# plots histogram
plotHistSmry = df1['WHYTRP1S'].hist(bins=25)
plotPieSmry = plt.pie(df1['WHYTRP1S'])
#plotPieSmry = plt.pie(df1['WHYTRP1S'], labels=whyIDsumList, autopct='%1.0f%%)
#plt.plot("whyDescSmry",type="bar")
#df1["WHYFROM"].plot(kind="bar")
#first5rows1['whyDescSmry'].hist()