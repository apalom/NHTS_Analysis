# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 11:25:54 2018

@author: Alex Palomino
"""

# import libraries
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import timeit

start_time = timeit.default_timer()

# filter dataframe zero (raw NHTS2009) to dimensions of interest
dfAlex = df0.filter(['TRAVDAY','STRTTIME','TRVL_MIN','WHYTRP1S'], axis = 1)

kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300).fit(dfAlex)

phi = kmeans.predict(dfAlex)

centers = kmeans.cluster_centers_

score = kmeans.score(dfAlex)

elapsed = timeit.default_timer() - start_time

# timeit statement
print('Execution time: {0:.4f} sec'.format(elapsed))