# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 09:28:58 2018

@author: Alex
"""

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt

clusters = 4

kMX = np.array([[0, 2], [2, 3], [2, 1], [1, 0], [2, 3],
              [4, 2], [1, 4], [3, 1], [4, 4], [3, 0]])

kM = KMeans(n_clusters=clusters, init='k-means++').fit(kMX)

kM_labels = kM.labels_

kM_predict = kM.predict(kMX)

kM_centers = kM.cluster_centers_

kM_score = kM.score(kMX)

kM_Metscore = metrics.adjusted_rand_score(kM_labels, kM_predict)


dst = np.zeros((10,clusters))
row = 0
col = 0
for pt in kMX:
    col = 0
    for c in kM_centers:
            dst[row][col] = distance.euclidean(pt,c)
            col += 1
    row += 1

idx = np.argmin(dst, axis = 0)
kM_centersReal = np.zeros((clusters,2))
j = 0

for i in idx:
    kM_centersReal[j] = kMX[i]
    j += 1
    
    #np.argmin(a, axis=0)


fig = plt.figure()
kMX_plot = plt.figure() 

#plt.grid(True, which='both')

plt.scatter(kMX[:,0],kMX[:,1])
plt.scatter(kM_centers[:,0],kM_centers[:,1],marker='o')
plt.scatter(kM_centersReal[:,0],kM_centersReal[:,1],c='w',marker='*',s=10)
print('k-Means')

