# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 11:25:54 2018

@author: Alex Palomino
"""

# import libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from scipy.spatial import distance
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import timeit

start_time = timeit.default_timer()

clusters = 4
numTrips = 1000
colNames = ['TRVLCMIN','HHFAMINC']
# Filter dataframe zero (raw NHTS2009) to dimensions of interest
#dfNHTS = df0.filter(['STRTTIME','TRVL_MIN','WHYTRP1S'], axis = 1)
dfNHTS = df0.filter(colNames, axis = 1)
dfNHTS = dfNHTS.head(numTrips) # shows first n rows

# Normalize dataframe by max value in each row
dfNorm = dfNHTS
dfNorm.iloc[:,0] = dfNHTS.iloc[:,0]/dfNHTS.max()[0]
dfNorm.iloc[:,1] = dfNHTS.iloc[:,1]/dfNHTS.max()[1]

# Calculate k-Means
kmeans = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfNHTS)

phi_true = kmeans.labels_
phi_predict = kmeans.predict(dfNorm)

centers = kmeans.cluster_centers_
score = kmeans.score(dfNorm)

# Compute Clustering Metrics
n_clusters_ = len(centers)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(phi_true, phi_predict))
print("Completeness: %0.3f" % metrics.completeness_score(phi_true, phi_predict))
print("V-measure: %0.3f" % metrics.v_measure_score(phi_true, phi_predict))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(phi_true, phi_predict))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(phi_true, phi_predict))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(dfNorm, phi_predict, metric='sqeuclidean'))

# timeit statement
elapsed = timeit.default_timer() - start_time
print('Execution time: {0:.4f} sec'.format(elapsed))

# %% 2-D Plot 1

rows = dfNorm.shape[0]

# Create matrix out of normalized dataframe
matNorm = dfNorm.as_matrix()

# Re-Create dfNHTS 
dfNHTS = df0.filter(colNames, axis = 1)
dfNHTS = dfNHTS.head(numTrips) # shows first n rows
matNHTS = dfNHTS.as_matrix()

# Find min distance between dfNorm domain and centers
dst = np.zeros((rows,clusters))
row = 0
col = 0
for pt in matNorm:
    col = 0
    for c in centers:
            dst[row][col] = distance.euclidean(pt,c)
            col += 1
    row += 1

idx = np.argmin(dst, axis = 0)
plotCentersReal = np.zeros((clusters,2))
centersReal = np.zeros((clusters,2))
j = 0

for i in idx:
    plotCentersReal[j] = matNorm[i]
    centersReal[j] = matNHTS[i]
    j += 1

 # %% Create centers dataframes

d = {colNames[0]: centersReal[:,0], colNames[1]: centersReal[:,1]}
dfCenters = pd.DataFrame(data=d)

# %% 2-D Plot 2

k1 = []
k2 = []
k3 = []
k4 = []
k5 = []

for j in range(0,len(dfNorm.index)):
    if phi_true[j] == 0:
        k1.append(j)
    if phi_true[j] == 1:
        k2.append(j)
    if phi_true[j] == 2:
        k3.append(j)
    if phi_true[j] == 3:
        k4.append(j)
    if phi_true[j] == 4:
        k5.append(j)

dfk1 = dfNorm.filter(k1, axis = 0)
dfk2 = dfNorm.filter(k2, axis = 0)
dfk3 = dfNorm.filter(k3, axis = 0)
dfk4 = dfNorm.filter(k4, axis = 0)
dfk5 = dfNorm.filter(k5, axis = 0)

plt.figure(figsize=(8,6))

for k in range(len(k1)):
    row = k1[k]
    plt.scatter(dfk1.loc[row][0],dfk1.loc[row][1],c='r',marker=".")

for k in range(len(k2)):
    row = k2[k]
    plt.scatter(dfk2.loc[row][0],dfk2.loc[row][1],c='g',marker=".")

for k in range(len(k3)):
    row = k3[k]
    plt.scatter(dfk3.loc[row][0],dfk3.loc[row][1],c='b',marker=".")

for k in range(len(k4)):
    row = k4[k]
    plt.scatter(dfk4.loc[row][0],dfk4.loc[row][1],c='m',marker=".")

for k in range(len(k5)):
    row = k5[k]
    plt.scatter(dfk5.loc[row][0],dfk5.loc[row][1],c='k',marker=".")

# Plot k-Mean Centers
plt.scatter(centers[:,0],centers[:,1],marker='v')

# Plot Real Centers (centers in domain)
plt.scatter(plotCentersReal[:,0],plotCentersReal[:,1],c='w',marker='*',s=5)

plt.title('2-D Cluster')
plt.xlabel(colNames[0])
plt.ylabel(colNames[1])


