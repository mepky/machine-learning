# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 06:42:07 2018

@author: praveen kumar
"""
#%reset -f

#importing the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset with pandas
dataset=pd.read_csv("C:/Users/praveen/Desktop/machine-learning udemy a-z/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering/K_Means/Mall_Customers.csv")
x=dataset.iloc[:,[3,4]].values

#using the elbow method to find the optimal numer of cluster
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("The elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("Wcss")
plt.show()

#Fitting kmeas to the mall dataset
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)

#Visulaizing the clusters
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='cluster 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='green',label='cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='blue',label='cluster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='cyan',label='cluster 4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='magenta',label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1],s=300,c='yellow',label='centroid')
plt.title("clusters of clients")
plt.xlabel("Annual income(k$)")
plt.ylabel("Spending salaries(1-100)")
plt.legend()
plt.show()






