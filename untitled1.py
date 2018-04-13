# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:47:55 2018

@author: avi_b
"""

from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
import random

file = open("C2.txt")
X=np.loadtxt(file)
X= np.delete(X,0,1)
#b=0.2
#random.uniform()
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print(kmeans)

kmeans = KMeans(n_clusters=3, init='random', n_init=2 ,max_iter=1, random_state=np.random).fit(X)
print(kmeans)
print('yessss')
print(np.random)
#Cen=[]
print(kmeans.cluster_centers_)

phi=kmeans.predict(X)
Cen1=kmeans.fit_predict(X)
Cen=kmeans.cluster_centers_
qq2=kmeans.fit(X)
qq3=kmeans.score(X)
qq4=kmeans.transform(X)
qq5=kmeans.fit_transform(X)
#kmeans.set_params=np.random.uniform()
#kmeans.algorithm(auto)
#random_state=np.random.uniform()  
print(kmeans)
X1=[]
X2=[]
X3=[]
g=0
for j in range(len(X)):
        if phi[j]==g:
            #print(j)
            X1.append(X[j])
            z1=np.array(X1)
            
        if phi[j]==g+1:
            X2.append(X[j])
            z2=np.array(X2)
        if phi[j]==g+2:
            X3.append(X[j])
            z3=np.array(X3)
 

plt.figure(figsize=(10, 5))           
plt.scatter(z1[:,0], z1[:,1], c='r')
plt.scatter(z2[:,0], z2[:,1], c='b')
plt.scatter(z3[:,0], z3[:,1], c='m')
plt.scatter(Cen[:,0], Cen[:,1], c='k', marker='*')
plt.show()
        

   


