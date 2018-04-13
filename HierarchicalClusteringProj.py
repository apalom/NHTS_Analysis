
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:34:32 2018

@author: avi_b
"""

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np


from scipy.cluster.hierarchy import fcluster
import pandas as pd
import timeit
import matplotlib.pyplot as plt

# initialize values   
start_time = timeit.default_timer()
whyID = {}
whyIDsum = {}
whyIDsumList = []

# NHTS2009 Data Location for Alex's Laptop 
#df0 = pd.read_csv(r'C:\Users\Avishan\Box\CS6140 Project\Data\CSV\DAYV2PUB.CSV', header=0)
#df0 = pd.read_csv(r'C:\Users\avi_b\Box\CS6140 Project\Data\CSV\DAYV2PUB.CSV', header=0)

df2 = pd.read_csv(r'C:\Users\avi_b\Box\Work\Second semester\Data mining\CS6140_Project\NHTS 2009\CA-Jan2009-filter.csv', header=0)

df1 = df2.filter(['STRTTIME','TRVL_MIN','ENDTIME','TRAVDAY','WHYTRP1S'], axis=1)

y1=[]
y2=[]
y3=[]
y4=[]

y1=np.array(df1)
y3=y1[:,[0,4]]
y4=y3[0:20000,0:2 ]
#print(y3)

y2=df1.iloc[:,0]
#print(y2)

np.set_printoptions(precision=5, suppress=True)


file = open("C1.txt")
X1=np.loadtxt(file)
#print(X)
#index=[1,2]
X1= np.delete(X1,0,1)

c=np.zeros((3,2))

X=y4
plt.scatter(X[:,0], X[:,1])
plt.show()
Z1= linkage(X, 'complete')
#print(Z1)

phi=fcluster(Z1, 9, criterion='maxclust')
#plt.scatter(phi,X[:,0], c='r')
#plt.show()


X1=[]
X2=[]
X3=[]
X4=[]
X5=[]
X6=[]
X7=[]
X8=[]
X9=[]
g=1


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
        if phi[j]==g+3:
            X4.append(X[j])
            z4=np.array(X4)
              
        if phi[j]==g+4:
            X5.append(X[j])
            z5=np.array(X5)
               
        if phi[j]==g+5:
            X6.append(X[j])
            z6=np.array(X6)
              
        if phi[j]==g+6:
            X7.append(X[j])
            z7=np.array(X7)
              
        if phi[j]==g+7:
            X8.append(X[j])
            z8=np.array(X8)
              
        if phi[j]==g+8:
            X9.append(X[j])
            z9=np.array(X9)
            
            
        
c[0]=(np.average(z1[:,0]), np.average(z1[:,1])) 
c[1]=(np.average(z2[:,0]), np.average(z2[:,1])) 
c[2]=(np.average(z3[:,0]), np.average(z3[:,1])) 
            
            
plt.figure(figsize=(6, 5))    
plt.title('Hierarchical Clustering-complete')
plt.xlabel('Start Time')
plt.ylabel('Purpos of Trip ')       
plt.scatter(z1[:,0], z1[:,1], c='r')
plt.scatter(z2[:,0], z2[:,1], c='b')
plt.scatter(z3[:,0], z3[:,1], c='g')
plt.scatter(z4[:,0], z4[:,1], c='m')
plt.scatter(z5[:,0], z5[:,1], c='c')
plt.scatter(z6[:,0], z6[:,1], c='k')
plt.scatter(z7[:,0], z7[:,1], c='y')
plt.scatter(z8[:,0], z8[:,1], c='b', marker='*')
plt.scatter(z9[:,0], z9[:,1], c='r', marker='+')


#plt.scatter(c[:,0], c[:,1], c='k', marker='*')
plt.show()


b1=set(Z1[:,3])
c1=len(b1)
print(len(b1))
idxs = [1, 2, 3,4]
#plt.scatter(X[:,0], X[:,1], c='r')
#plt.scatter(X[idxs,0], X[idxs,1], c='y')
#plt.scatter(Z1[:,2], Z1[:,3], c='r')
#plt.show()

#%%
#plt.figure(figsize=(10, 5))
#plt.title('Hierarchical Clustering Dendrogram')
#plt.xlabel('sample index')
#plt.ylabel('distance')
#dendrogram(
#    Z1,
#    leaf_rotation=90.,  # rotates the x axis labels
#    leaf_font_size=8.,  # font size for the x axis labels
#)
#plt.show()


#plt.title('Hierarchical Clustering Dendrogram (truncated)')
#plt.xlabel('sample index or (cluster size)')
#plt.ylabel('distance')
#dendrogram(
#    Z1,
#    truncate_mode='lastp',  # show only the last p merged clusters
#    p=4,  # show only the last p merged clusters
#    leaf_rotation=90.,
#    leaf_font_size=12.,
#    show_contracted=True,  # to get a distribution impression in truncated branches
#)
#plt.show()



