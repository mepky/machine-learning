# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 22:08:26 2018

@author: praveen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('C:/Users/praveen/Desktop/machine-learning udemy a-z/Machine Learning A-Z Template Folder\Part 2 - Regression/Section 4 - Simple Linear Regression/Simple_Linear_Regression/salary_data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values
x=pd.DataFrame(x)
y=pd.DataFrame(y)

#spliting the data set into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#to defining regressor
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#prediction of salary on x_test data
y_pred=regressor.predict(x_test)
y1_pred=regressor.predict(x_train)

#visulazing the  training set  result
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,y1_pred,color='blue')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.title('salary vs experience (training set)')
plt.show()