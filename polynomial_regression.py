# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 23:32:24 2018

@author: praveen
"""
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset=pd.read_csv('C:/Users/praveen/Desktop/machine-learning udemy a-z/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Polynomial_Regression/Position_Salaries.csv')
x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values

#fitting linearregression to the dataset
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x,y)

#Fitting polynomial regression to the  dataset 
from sklearn.preprocessing import PolynomialFeatures
pol_reg=PolynomialFeatures(degree=5)
x_poly=pol_reg.fit_transform(x)
linear_reg_2=LinearRegression()
linear_reg_2.fit(x_poly,y)

#visualizing the linear regression
plt.scatter(x,y,color='red')
plt.plot(x,linear_reg.predict(x),color='blue')
plt.title('truth or bluf (linear regression)')
plt.xlabel('position value')
plt.ylabel('salary')
plt.show()

#visualizing the polynomial regression
plt.scatter(x,y,color='red')
plt.plot(x,linear_reg_2.predict(pol_reg.fit_transform(x)),color='blue')
plt.title('truth or bluf (polynomial regression)')
plt.xlabel('position value')
plt.ylabel('salary')
plt.show()

#predicting a new result with linear regression
linear_reg.predict(6.5)
#predicting a new result with polynomial regression

linear_reg_2.predict(pol_reg.fit_transform(6.5))






