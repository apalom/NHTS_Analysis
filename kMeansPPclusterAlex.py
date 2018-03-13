# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 11:25:54 2018

@author: Alex Palomino
"""

# import libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import timeit

start_time = timeit.default_timer()

# filter dataframe zero (raw NHTS2009) to dimensions of interest
dfAlex = df0.filter(['STRTTIME','TRVL_MIN','WHYTRP1S'], axis = 1)

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300).fit(dfAlex)


phi_predict = kmeans.predict(dfAlex)
phi_true = kmeans.labels_
centers = kmeans.cluster_centers_
score = kmeans.score(dfAlex)


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
#print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(dfAlex, phi_predict, metric='sqeuclidean'))

# timeit statement
elapsed = timeit.default_timer() - start_time
print('Execution time: {0:.4f} sec'.format(elapsed))

# %% 3-D Plot

k1 = []
k2 = []
k3 = []
k4 = []
k5 = []

for j in range(0,len(dfAlex.index)):
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

dfk1 = dfAlex.filter(k1, axis = 0)
dfk2 = dfAlex.filter(k2, axis = 0)
dfk3 = dfAlex.filter(k3, axis = 0)
dfk4 = dfAlex.filter(k4, axis = 0)
dfk5 = dfAlex.filter(k5, axis = 0)

fig = plt.figure()
threedee = plt.figure().gca(projection='3d')

plt.grid(True, which='both')

for k in range(len(k1)):
    row = k1[k]
    threedee.scatter(dfk1.loc[row][0],dfk1.loc[row][1],dfk1.loc[row][2],c='r',marker=".")

for k in range(len(k2)):
    row = k2[k]
    threedee.scatter(dfk2.loc[row][0],dfk2.loc[row][1],dfk2.loc[row][2],c='g',marker=".")

for k in range(len(k3)):
    row = k3[k]
    threedee.scatter(dfk3.loc[row][0],dfk3.loc[row][1],dfk3.loc[row][2],c='b',marker=".")

for k in range(len(k4)):
    row = k4[k]
    threedee.scatter(dfk4.loc[row][0],dfk4.loc[row][1],dfk4.loc[row][2],c='m',marker=".")

for k in range(len(k5)):
    row = k5[k]
    threedee.scatter(dfk5.loc[row][0],dfk5.loc[row][1],dfk5.loc[row][2],c='k',marker=".")

threedee.set_xlabel('STRTTIME')
threedee.set_ylabel('TRVL_MIN')
threedee.set_zlabel('WHYTRP1S')

plt.show()
