# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 11:25:54 2018

@author: Alex Palomino
"""

# import libraries
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import timeit

start_time = timeit.default_timer()

# filter dataframe zero (raw NHTS2009) to dimensions of interest
dfAlex = df0.filter(['STRTTIME','TRVL_MIN','WHYTRP1S'], axis = 1)

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300).fit(dfAlex)

phi = kmeans.predict(dfAlex)

centers = kmeans.cluster_centers_

score = kmeans.score(dfAlex)

elapsed = timeit.default_timer() - start_time

# timeit statement
print('Execution time: {0:.4f} sec'.format(elapsed))

# %% 3-D Plot

k1 = []
k2 = []
k3 = []
k4 = []
k5 = []

for j in range(0,len(dfAlex.index)):
    if phi[j] == 0:
        k1.append(j)
    if phi[j] == 1:
        k2.append(j)
    if phi[j] == 2:
        k3.append(j)
    if phi[j] == 3:
        k4.append(j)
    if phi[j] == 4:
        k5.append(j)

k1 = np.asarray(k1)
k2 = np.asarray(k2)
k3 = np.asarray(k3)
k4 = np.asarray(k4)
k5 = np.asarray(k5)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.grid(True, which='both')
#plt.axvline(x=0, color='k')
#plt.axhline(y=0, color='k')

for k in range(len(k1)):
    plt.scatter(dfAlex[k1[k]], c='r',marker=".")

plt.scatter(dfAlex[k2,0], dfAlex[k2,1], dfAlex[k1,2], c='g',marker=".")
plt.scatter(dfAlex[k3,0], dfAlex[k3,1], dfAlex[k1,2], c='b',marker=".")
plt.scatter(dfAlex[k4,0], dfAlex[k4,1], dfAlex[k4,2], c='m',marker=".")
plt.scatter(dfAlex[k4,0], dfAlex[k4,1], dfAlex[k4,2], c='k',marker=".")

'''
x3 = dfAlex['STRTTIME']
y3 = dfAlex['TRVL_MIN']
z3 = dfAlex['WHYTRP1S']

ax.scatter(x3, y3, z3)
'''

ax.set_xlabel('STRTTIME')
ax.set_ylabel('TRVL_MIN')
ax.set_zlabel('WHYTRP1S')

plt.show()
