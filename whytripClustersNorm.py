# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:45:02 2018

@author: Alex Palomino
"""

# import libraries
import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance

elapsed = 0
timeExec= []
start_time = timeit.default_timer()

# Variables
clusters = 3
numTrips = 50000

# NHTS2017 Data Location for Alex's Lab Computer
df0 = pd.read_csv(r'C:\Users\Alex\Documents\NHTS_2017\trippub.CSV', header=0)
firstRAWrows = df0.head(50)

# Column names for filter
#-- Start vs. End --#
#colNames = ['STRTTIME','ENDTIME','WHYTRP1S']
#dfFilter = ['STRTTIME','ENDTIME']
#-- End vs. Dwell --#
colNames = ['ENDTIME','DWELTIME','WHYTRP1S']
dfFilter = ['ENDTIME','DWELTIME']
#-- Start vs. Trip Miles --#
#colNames = ['STRTTIME','TRPMILES','WHYTRP1S']
#dfFilter = ['STRTTIME','TRPMILES']

dfNHTS = df0.filter(colNames, axis=1)
dfNHTS = dfNHTS.head(numTrips)
firstNHTSrows = dfNHTS.head(50)   

'''
# Convert Start Times to minutes
idx = 0
for idx, row in dfNHTS.iterrows():
    print(idx)
    if len(str(dfNHTS['STRTTIME'][idx])) == 4:
        hr = int(str(dfNHTS['STRTTIME'][idx])[0] + str(dfNHTS['STRTTIME'][idx])[1])
        hr = hr * 60
        minute = int(str(dfNHTS['STRTTIME'][idx])[2] + str(dfNHTS['STRTTIME'][idx])[3])
        timeMinutes = hr + minute
        dfNHTS['STRTTIME'][idx] = timeMinutes
    elif len(str(dfNHTS['STRTTIME'][idx])) == 3:
        hr = int(str(dfNHTS['STRTTIME'][idx])[0])
        hr = hr * 60
        minute = int(str(dfNHTS['STRTTIME'][idx])[1] + str(dfNHTS['STRTTIME'][idx])[2])
        timeMinutes = hr + minute
        dfNHTS['STRTTIME'][idx] = timeMinutes
'''
# Convert End Times to minutes
idx = 0
for idx, row in dfNHTS.iterrows():
    print(idx)
    if len(str(dfNHTS['ENDTIME'][idx])) == 4:
        hr = int(str(dfNHTS['ENDTIME'][idx])[0] + str(dfNHTS['ENDTIME'][idx])[1])
        hr = hr * 60
        minute = int(str(dfNHTS['ENDTIME'][idx])[2] + str(dfNHTS['ENDTIME'][idx])[3])
        timeMinutes = hr + minute
        dfNHTS['ENDTIME'][idx] = timeMinutes
    elif len(str(dfNHTS['ENDTIME'][idx])) == 3:
        hr = int(str(dfNHTS['ENDTIME'][idx])[0])
        hr = hr * 60
        minute = int(str(dfNHTS['ENDTIME'][idx])[1] + str(dfNHTS['ENDTIME'][idx])[2])
        timeMinutes = hr + minute
        dfNHTS['ENDTIME'][idx] = timeMinutes


# For START v. END time normalization
#dfNHTS.iloc[:,0] = dfNHTS.iloc[:,0]/dfNHTS.max()[0] #Normalize STRTTIME
#dfNHTS.iloc[:,1] = dfNHTS.iloc[:,1]/dfNHTS.max()[1] #Normalize ENDTIME
        
dfNHTS.iloc[:,0] = dfNHTS.iloc[:,0]/dfNHTS.max()[0] #Normalize ENDTIME
dfNHTS.iloc[:,1] = dfNHTS.iloc[:,1]/dfNHTS.max()[1] #Normalize DWELTIME


'''
# For START v. END time normalization
dfNHTS.iloc[:,0] = dfNHTS.iloc[:,0]/2400 #Normalize STRTTIME
dfNHTS.iloc[:,1] = dfNHTS.iloc[:,1]/2400 #Normalize ENDTIME

# For END v. DWELL time normalization
dfNHTS.iloc[:,0] = dfNHTS.iloc[:,0]/2400 #Normalize ENDTIME
dfNHTS.iloc[:,1] = dfNHTS.iloc[:,1]/dfNHTS.max()[1] #Normalize DWELTIME

dfNHTS.iloc[:,0] = dfNHTS.iloc[:,0]/2400 #Normalize STRTTIME
dfNHTS.iloc[:,1] = dfNHTS.iloc[:,1]/dfNHTS.max()[1] #Normalize TRPMILES
'''

firstNHTSrows = dfNHTS.head(50) 

# Home WHYTRP1S 
dfHome = dfNHTS.loc[(dfNHTS['WHYTRP1S'] == 1) & (dfNHTS['DWELTIME'] > 0)]
dfHome = dfHome.filter(dfFilter)
dfHome = dfHome.head(numTrips)
dfHome = dfHome.reset_index()
dfHome = dfHome.drop(['index'], axis=1)

kHome = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfHome)
kHomePhi = kHome.labels_
kHomePhiPredict = kHome.predict(dfHome)
kHomeCenters = kHome.cluster_centers_
#['ENDTIME','DWELTIME']
#dfWork = dfNHTS.loc[(dfNHTS['WHYTRP1S'] == 10) & (dfNHTS['HHSTFIPS'] == 6)]

'''
# Work WHYTRP1S 
dfWork = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 10]
dfWork = dfWork.filter(dfFilter)
dfWork = dfWork.head(numTrips)
dfWork = dfWork.reset_index()
dfWork = dfWork.drop(['index'], axis=1)

kWork = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfWork)
kWorkPhi = kWork.labels_
kWorkPhiPredict = kWork.predict(dfWork)
kWorkCenters = kWork.cluster_centers_


# School WHYTRP1S 
dfSchool = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 20]
dfSchool = dfSchool.filter(dfFilter)
dfSchool = dfSchool.head(numTrips)
dfSchool = dfSchool.reset_index()
dfSchool = dfSchool.drop(['index'], axis=1)

kSchool = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfSchool)
kSchoolPhi = kSchool.labels_
kSchoolCenters = kSchool.cluster_centers_


# Medical WHYTRP1S 
dfMedical = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 30]
dfMedical = dfMedical.filter(['ENDTIME','DWELTIME'])
dfMedical = dfMedical.head(5000)
dfMedical = dfMedical.reset_index()
dfMedical = dfMedical.drop(['index'], axis=1)

kMedical = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfMedical)
kMedicalPhi = kMedical.labels_
kMedicalCenters = kMedical.cluster_centers_

# Shopping WHYTRP1S 
dfShopping = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 40]
dfShopping = dfShopping.filter(['ENDTIME','DWELTIME'])
dfShopping = dfShopping.head(5000)
dfShopping = dfShopping.reset_index()
dfShopping = dfShopping.drop(['index'], axis=1)

kShopping = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfShopping)
kShoppingPhi = kShopping.labels_
kShoppingCenters = kShopping.cluster_centers_


# Social WHYTRP1S 
dfSocial = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 50]
dfSocial = dfSocial.filter(['ENDTIME','DWELTIME'])
dfSocial = dfSocial.head(5000)
dfSocial = dfSocial.reset_index()
dfSocial = dfSocial.drop(['index'], axis=1)

kSocial = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfSocial)
kSocialPhi = kSocial.labels_
kSocialCenters = kSocial.cluster_centers_

# Transport WHYTRP1S 
dfTransport = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 70]
dfTransport = dfTransport.filter(['ENDTIME','DWELTIME'])
dfTransport = dfTransport.head(5000)
dfTransport = dfTransport.reset_index()
dfTransport = dfTransport.drop(['index'], axis=1)

kTransport = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfTransport)
kTransportPhi = kTransport.labels_
kTransportCenters = kTransport.cluster_centers_


# Meals WHYTRP1S 
dfMeals = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 80]
dfMeals = dfMeals.filter(['ENDTIME','DWELTIME'])
dfMeals = dfMeals.head(5000)
dfMeals = dfMeals.reset_index()
dfMeals = dfMeals.drop(['index'], axis=1)

kMeals = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfMeals)
kMealsPhi = kMeals.labels_
kMealsCenters = kMeals.cluster_centers_

# Other WHYTRP1S 
dfOther = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 97]
dfOther = dfOther.filter(['STRTTIME','ENDTIME'])
dfOther = dfOther.head(5000)
dfOther = dfOther.reset_index()
dfOther = dfOther.drop(['index'], axis=1)   

kOther = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfOther)
kOtherPhi = kOther.labels_    
kOtherCenters = kOther.cluster_centers_
'''

elapsed = timeit.default_timer() - start_time

print('Step 1 Execution time: {0:.2f} sec'.format(elapsed))

# %% Calculate Real Centers

elapsed = 0
timeExec= []
start_time = timeit.default_timer()

dfPlot = dfHome
matPlot = dfPlot.as_matrix()
phi_true = kHomePhi
centers = kHomeCenters

rows  = dfPlot.shape[0]

# Find min distance between data domain and centers
dst = np.zeros((rows,clusters))
row = 0
col = 0
for pt in matPlot:
    col = 0
    for c in centers:
            dst[row][col] = distance.euclidean(pt,c)
            col += 1
    row += 1

idx = np.argmin(dst, axis = 0)
centersReal = np.zeros((clusters,2))
j = 0

for i in idx:
    centersReal[j] = matPlot[i]
    j += 1

elapsed = timeit.default_timer() - start_time
print('Step 2 Execution time: {0:.2f} sec'.format(elapsed))
   
# %% Plot Clusters
 
elapsed = 0
timeExec= []
start_time = timeit.default_timer()

k1 = []
k2 = []
k3 = []
k4 = []
k5 = []
k6 = []
k7 = []
k8 = []
k9 = []


for j in range(0,len(dfPlot.index)):
    if phi_true[j] == 0 and clusters > 0:
        k1.append(j)
    if phi_true[j] == 1 and clusters > 1:
        k2.append(j)
    if phi_true[j] == 2 and clusters > 2:
        k3.append(j)
    if phi_true[j] == 3 and clusters > 3:
        k4.append(j)
    if phi_true[j] == 4 and clusters > 4:
        k5.append(j)
    if phi_true[j] == 5 and clusters > 5:
        k6.append(j)
    if phi_true[j] == 6 and clusters > 6:
        k7.append(j)
    if phi_true[j] == 7 and clusters > 7:
        k8.append(j)
    if phi_true[j] == 8 and clusters > 8:
        k9.append(j)

if clusters > 0:
    dfK1 = dfPlot.filter(k1, axis = 0)
if clusters > 1:    
    dfK2 = dfPlot.filter(k2, axis = 0)
if clusters > 2:
    dfK3 = dfPlot.filter(k3, axis = 0)
if clusters > 3:    
    dfK4 = dfPlot.filter(k4, axis = 0)
if clusters > 4:
    dfK5 = dfPlot.filter(k5, axis = 0)
if clusters > 5:
    dfK6 = dfPlot.filter(k6, axis = 0)
if clusters > 6:
    dfK7 = dfPlot.filter(k7, axis = 0)
if clusters > 7:
    dfK8 = dfPlot.filter(k8, axis = 0)
if clusters > 8:
    dfK9 = dfPlot.filter(k9, axis = 0)

plt.figure(figsize=(8,6))

if clusters > 0:
    for k in range(len(dfK1)):
        row = dfK1.index[k]
        plt.scatter(dfK1.loc[row][0],dfK1.loc[row][1],c='b',marker=".")

    print('Cluster - 1')

if clusters > 1:
    for k in range(len(dfK2)):
        row = dfK2.index[k]
        plt.scatter(dfK2.loc[row][0],dfK2.loc[row][1],c='g',marker=".")
    
    print('Cluster - - 2')

if clusters > 2:    
    for k in range(len(dfK3)):
        row = dfK3.index[k]
        plt.scatter(dfK3.loc[row][0],dfK3.loc[row][1],c='r',marker=".")
    
    print('Cluster - - - 3')
    
if clusters > 3:    
    for k in range(len(dfK4)):
        row = dfK4.index[k]
        plt.scatter(dfK4.loc[row][0],dfK4.loc[row][1],c='c',marker=".")
    
    print('Cluster - - - - 4')
    
if clusters > 4:    
    for k in range(len(dfK5)):
        row = dfK5.index[k]
        plt.scatter(dfK5.loc[row][0],dfK5.loc[row][1],c='m',marker=".")
    
    print('Cluster - - - - - 5')
    
if clusters > 5:    
    for k in range(len(dfK6)):
        row = dfK6.index[k]
        plt.scatter(dfK6.loc[row][0],dfK6.loc[row][1],c='y',marker=".")
    
    print('Cluster - - - - - - 6')

if clusters > 6:    
    for k in range(len(dfK7)):
        row = dfK7.index[k]
        plt.scatter(dfK7.loc[row][0],dfK7.loc[row][1],c='y',marker=".")
    
    print('Cluster - - - - - - - 7')

if clusters > 7:    
    for k in range(len(dfK8)):
        row = dfK8.index[k]
        plt.scatter(dfK8.loc[row][0],dfK8.loc[row][1],c='y',marker=".")
    
    print('Cluster - - - - - - - - 8')

if clusters > 8:    
    for k in range(len(dfK9)):
        row = dfK9.index[k]
        plt.scatter(dfK9.loc[row][0],dfK9.loc[row][1],c='y',marker=".")
    
    print('Cluster - - - - - - - - - 9')  
  
# Plot k-Mean Centers
plt.scatter(centers[:,0],centers[:,1],marker='v')

# Plot Real Centers (centers in domain)
plt.scatter(centersReal[:,0],centersReal[:,1],c='w',marker='*',s=5)

plt.title('Home 2-D Clusters: Military Time ')
plt.xlabel(colNames[0])
plt.ylabel(colNames[1])
plt.show()

elapsed = timeit.default_timer() - start_time
print('Step 3 Execution time: {0:.2f} sec'.format(elapsed))

   
# %% Calculate Distance Score

elapsed = 0
timeExec= []
start_time = timeit.default_timer()

rows  = dfPlot.shape[0]

# Distance scores between clusters and centers
dstAvg = np.zeros((1,clusters))
dstMin = np.zeros((1,clusters))
dstMax = np.zeros((1,clusters))

if clusters > 0:    
    matK1 = dfK1.as_matrix()
    dst1 = np.zeros((len(dfK1),1))
    row = 0
    for pt in matK1:
        dst1[row] = distance.euclidean(pt,centersReal[0])
        row += 1
        
if clusters > 1:    
    matK2 = dfK2.as_matrix()
    dst2 = np.zeros((len(dfK2),1))   
    row = 0
    for pt in matK2:
        dst2[row] = distance.euclidean(pt,centersReal[1])
        row += 1

if clusters > 2: 
    matK3 = dfK3.as_matrix()
    dst3 = np.zeros((len(dfK3),1))    
    row = 0   
    for pt in matK3:
        dst3[row] = distance.euclidean(pt,centersReal[2])
        row += 1

if clusters > 3:
    matK4 = dfK4.as_matrix()
    dst4 = np.zeros((len(dfK4),1))    
    row = 0    
    for pt in matK4:
        dst4[row] = distance.euclidean(pt,centersReal[3])
        row += 1

if clusters > 4: 
    matK5 = dfK5.as_matrix()
    dst5 = np.zeros((len(dfK5),1))
    row = 0    
    for pt in matK5:
        dst5[row] = distance.euclidean(pt,centersReal[4])
        row += 1

if clusters > 5:    
    matK6 = dfK6.as_matrix()
    dst6 = np.zeros((len(dfK6),1))    
    row = 0   
    for pt in matK6:
        dst6[row] = distance.euclidean(pt,centersReal[5])
        row += 1

if clusters > 6:
    matK7 = dfK7.as_matrix()
    dst7 = np.zeros((len(dfK7),1))        
    row = 0   
    for pt in matK7:
        dst7[row] = distance.euclidean(pt,centersReal[6])
        row += 1

if clusters > 7:
    matK8 = dfK8.as_matrix()
    dst8 = np.zeros((len(dfK8),1))    
    row = 0   
    for pt in matK8:
        dst8[row] = distance.euclidean(pt,centersReal[7])
        row += 1

if clusters > 8:        
    matK9 = dfK9.as_matrix()
    dst9 = np.zeros((len(dfK9),1))    
    row = 0   
    for pt in matK9:
        dst9[row] = distance.euclidean(pt,centersReal[8])
        row += 1

# Calculate distance between cluster points and real center
if clusters > 0:  
    dstAvg[0][0] = np.mean(dst1)  
    dstMax[0][0] = np.max(dst1)  
    dstMin[0][0] = np.min(dst1[np.nonzero(dst1)])

if clusters > 1:    
    dstAvg[0][1] = np.mean(dst2)  
    dstMax[0][1] = np.max(dst2)
    dstMin[0][1] = np.min(dst2[np.nonzero(dst2)])

if clusters > 2:    
    dstAvg[0][2] = np.mean(dst3)
    dstMax[0][2] = np.max(dst3)  
    dstMin[0][2] = np.min(dst3[np.nonzero(dst3)])

if clusters > 3:    
    dstAvg[0][3] = np.mean(dst4)  
    dstMax[0][3] = np.max(dst4) 
    dstMin[0][3] = np.min(dst4[np.nonzero(dst4)])
    
if clusters > 4:    
    dstAvg[0][4] = np.mean(dst5)  
    dstMax[0][4] = np.max(dst5)  
    dstMin[0][4] = np.min(dst5[np.nonzero(dst5)])

if clusters > 5:    
    dstAvg[0][5] = np.mean(dst6)  
    dstMax[0][5] = np.max(dst6)  
    dstMin[0][5] = np.min(dst6[np.nonzero(dst6)])
    
if clusters > 6:    
    dstAvg[0][6] = np.mean(dst7) 
    dstMax[0][6] = np.max(dst7)
    dstMin[0][6] = np.min(dst7[np.nonzero(dst7)])

if clusters > 7:    
    dstAvg[0][7] = np.mean(dst8)  
    dstMax[0][7] = np.max(dst8) 
    dstMin[0][7] = np.min(dst8[np.nonzero(dst8)])

if clusters > 8:    
    dstAvg[0][8] = np.mean(dst9)  
    dstMax[0][8] = np.max(dst9) 
    dstMin[0][8] = np.min(dst9[np.nonzero(dst9)])

scoreAvg = np.mean(dstAvg)
scoreMax = np.mean(dstMax)
scoreMin = np.mean(dstMin)

elapsed = timeit.default_timer() - start_time
print('Step 4 Execution time: {0:.2f} sec'.format(elapsed))

   
# %% Plot Histograms
'''
elapsed = 0
timeExec= []
start_time = timeit.default_timer()

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import operator

histFilter = ['STRTTIME','ENDTIME','DWELTIME','TRPMILES','WHYTRP1S']

# Organize data for histogram
dfHist = df0.filter(histFilter, axis=1)
dfHist = dfHist.loc[(df0['WHYTRP1S'] == 10) & (df0['DWELTIME'] != -9)]
dfHist = dfHist.filter(histFilter)
dfHist = dfHist.head(numTrips)
dfHist = dfHist.reset_index()
dfHist = dfHist.drop(['index'], axis=1)
matHist = dfHist.as_matrix()

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

#-----------------------------------------------------------#
# Plot Starttime Histogram
plotHist0 = plt.hist(matHist[:,0], bins=24, normed='density')

# Create the formatter using the function to_percent. This multiplies all the
# default labels by 100, making them all percentages
scale = 10000
formatter = FuncFormatter(to_percent)

# Set the formatter
plt.yticks(np.arange(0, 0.00275, step=0.00025))
plt.gca().yaxis.set_major_formatter(formatter)
plt.title('Histogram of Start Time')
plt.show()

#----------------------------------------------------------#
# Plot Endtime Histogram
plotHist1 = plt.hist(matHist[:,1], bins=24, normed='density')

# Create the formatter using the function to_percent. This multiplies all the
# default labels by 100, making them all percentages
scale = 10000
formatter = FuncFormatter(to_percent)

# Set the formatter
plt.yticks(np.arange(0, 0.00275, step=0.00025))
plt.gca().yaxis.set_major_formatter(formatter)
plt.title('Histogram of Endtime')
plt.show()

# dfNHTS.iloc[:,1] = dfNHTS.iloc[:,1]/dfNHTS.max()[1] 

#----------------------------------------------------------#
# Plot Dwelltime Histogram
plotHist2 = plt.hist(matHist[:,2], bins=10, normed='density')

# Create the formatter using the function to_percent. This multiplies all the
# default labels by 100, making them all percentages
scale = 10000
formatter = FuncFormatter(to_percent)

# Set the formatter
plt.gca().yaxis.set_major_formatter(formatter)
plt.title('Histogram of Dwelltime')
plt.show()

#----------------------------------------------------------#
# Plot TripMiles Histogram
plotHist3 = plt.hist(matHist[:,3], bins='auto', normed='density', range=(0,50))
ax = plotHist3

# Create the formatter using the function to_percent. This multiplies all the
# default labels by 100, making them all percentages
#formatter = FuncFormatter(to_percent)

# Set the formatter
scale = 100
formatter = FuncFormatter(to_percent)

#plt.yticks(range(0, ymax, ymin))
plt.gca().yaxis.set_major_formatter(formatter)
plt.title('Histogram of Tripmiles')
plt.show()

#----------------------------------------------------------#
# Organize data for Trip Type histogram
dfTripType = df0.filter(histFilter, axis=1)
dfTripType = dfTripType.filter(['WHYTRP1S'])
dfTripType = dfTripType.head(numTrips)
matTripType = dfTripType.as_matrix()

pctHome = len(dfTripType.loc[dfNHTS['WHYTRP1S'] == 1])/numTrips
pctWork = len(dfTripType.loc[dfNHTS['WHYTRP1S'] == 10])/numTrips
pctSchool = len(dfTripType.loc[dfNHTS['WHYTRP1S'] == 20])/numTrips
pctMedical = len(dfTripType.loc[dfNHTS['WHYTRP1S'] == 30])/numTrips
pctShopping = len(dfTripType.loc[dfNHTS['WHYTRP1S'] == 40])/numTrips
pctSocial = len(dfTripType.loc[dfNHTS['WHYTRP1S'] == 50])/numTrips
pctTransport = len(dfTripType.loc[dfNHTS['WHYTRP1S'] == 70])/numTrips
pctMeals = len(dfTripType.loc[dfNHTS['WHYTRP1S'] == 80])/numTrips
pctOther = len(dfTripType.loc[dfNHTS['WHYTRP1S'] == 97])/numTrips

pctAll = [pctHome,pctWork,pctSchool,pctMedical,pctShopping,pctSocial,pctTransport,pctMeals,pctOther]

binNames = ['Home','Work','School','Medical','Shopping','Social','Transport','Meals','Other']

pctTypesD = {binNames[0]:pctAll[0],
            binNames[1]:pctAll[1],
            binNames[2]:pctAll[2],
            binNames[3]:pctAll[3],
            binNames[4]:pctAll[4],
            binNames[5]:pctAll[5],
            binNames[6]:pctAll[6],
            binNames[7]:pctAll[7],
            binNames[8]:pctAll[8]}

pctTypesD = dict(sorted(pctTypesD.items(), key=operator.itemgetter(1), reverse=True))

pos = np.arange(len(pctTypesD))

plt.bar(pos, pctTypesD.values(), align='center', alpha=0.5)
plt.xticks(pos, pctTypesD.keys(), rotation=70)


ymax = max(pctTypesD.values()) + 0.05
plt.ylim(0, ymax)

# Format y-axis as percentages
formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)

plt.title('Histogram of Trip Types')
plt.show()

elapsed = timeit.default_timer() - start_time
print('Step 5 Execution time: {0:.2f} sec'.format(elapsed))
'''