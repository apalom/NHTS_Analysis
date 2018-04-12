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

# Column names for filter
colNames = ['STRTTIME','ENDTIME','WHYTRP1S']

# NHTS2009 Data Location for Alex's Lab Computer
df0 = pd.read_csv(r'C:\Users\Alex\Documents\NHTS_2017\trippub.CSV', header=0)

dfNHTS = df0.filter(colNames, axis = 1)
firstNHTSrows = dfNHTS.head(50) 

# Variables
clusters = 4

# Home WHYTRP1S 
dfHome = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 1]
dfHome = dfHome.filter(['STRTTIME','ENDTIME'])
dfHome = dfHome.head(1000)
dfHome = dfHome.reset_index()
dfHome = dfHome.drop(['index'], axis=1)

kHome = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfHome)
kHomePhi = kHome.labels_
kHomeCenters = kHome.cluster_centers_

# Work WHYTRP1S 
dfWork = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 10]
dfWork = dfWork.filter(['STRTTIME','ENDTIME'])

kWork = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfWork)
kWorkPhi = kWork.labels_
kWorkCenters = kWork.cluster_centers_

# School WHYTRP1S 
dfSchool = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 20]
dfSchool = dfSchool.filter(['STRTTIME','ENDTIME'])

kSchool = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfSchool)
kSchoolPhi = kSchool.labels_
kSchoolCenters = kSchool.cluster_centers_

# Medical WHYTRP1S 
dfMedical = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 30]
dfMedical = dfMedical.filter(['STRTTIME','ENDTIME'])

kMedical = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfMedical)
kMedicalPhi = kMedical.labels_
kMedicalCenters = kMedical.cluster_centers_

# Shopping WHYTRP1S 
dfShopping = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 40]
dfShopping = dfShopping.filter(['STRTTIME','ENDTIME'])

kShopping = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfShopping)
kShoppingPhi = kShopping.labels_
kShoppingCenters = kShopping.cluster_centers_

# Social WHYTRP1S 
dfSocial = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 50]
dfSocial = dfSocial.filter(['STRTTIME','ENDTIME'])

kSocial = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfSocial)
kSocialPhi = kSocial.labels_
kSocialCenters = kSocial.cluster_centers_

# Transport WHYTRP1S 
dfTransport = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 70]
dfTransport = dfTransport.filter(['STRTTIME','ENDTIME'])

kTransport = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfTransport)
kTransportPhi = kTransport.labels_
kTransportCenters = kTransport.cluster_centers_

# Meals WHYTRP1S 
dfMeals = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 80]
dfMeals = dfMeals.filter(['STRTTIME','ENDTIME'])

kMeals = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfMeals)
kMealsPhi = kMeals.labels_
kMealsCenters = kMeals.cluster_centers_

# Other WHYTRP1S 
dfOther = dfNHTS.loc[dfNHTS['WHYTRP1S'] == 97]
dfOther = dfOther.filter(['STRTTIME','ENDTIME'])
   
kOther = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfOther)
kOtherPhi = kOther.labels_    
kOtherCenters = kOther.cluster_centers_

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

for j in range(0,len(dfPlot.index)):
    if phi_true[j] == 0:
        k1.append(j)
    if phi_true[j] == 1:
        k2.append(j)
    if phi_true[j] == 2:
        k3.append(j)
    if phi_true[j] == 3:
        k4.append(j)

dfK1 = dfPlot.filter(k1, axis = 0)
dfK2 = dfPlot.filter(k2, axis = 0)
dfK3 = dfPlot.filter(k3, axis = 0)
dfK4 = dfPlot.filter(k4, axis = 0)


plt.figure(figsize=(8,6))
       
for k in range(len(dfK1)):
    row = dfK1.index[k]
    plt.scatter(dfK1.loc[row][0],dfK1.loc[row][1],c='r',marker=".")

print('Flag --- 5')

for k in range(len(dfK2)):
    row = dfK2.index[k]
    plt.scatter(dfK2.loc[row][0],dfK2.loc[row][1],c='g',marker=".")

for k in range(len(dfK3)):
    row = dfK3.index[k]
    plt.scatter(dfK3.loc[row][0],dfK3.loc[row][1],c='b',marker=".")

for k in range(len(dfK4)):
    row = dfK4.index[k]
    plt.scatter(dfK4.loc[row][0],dfK4.loc[row][1],c='m',marker=".")
    
# Plot k-Mean Centers
plt.scatter(centers[:,0],centers[:,1],marker='v')

# Plot Real Centers (centers in domain)
plt.scatter(centersReal[:,0],centersReal[:,1],c='w',marker='*',s=5)

plt.title('2-D Cluster')
plt.xlabel(colNames[0])
plt.ylabel(colNames[1])


print('Step 3 Execution time: {0:.2f} sec'.format(elapsed))