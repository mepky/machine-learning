# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 22:06:27 2018

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

#Using the dendogram to find the maximum number of cluster
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))

plt.title("Dendograms")
plt.xlabel("customers")
plt.ylabel("Eucldian distance")
plt.show()