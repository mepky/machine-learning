# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 19:42:14 2018

@author: praveen
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset=pd.read_csv('C:/Users/praveen/Desktop/machine-learning udemy a-z/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Polynomial_Regression/Position_Salaries.csv')
x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values

#fitting linearregression to the dataset
'''from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x,y)'''

#Fitting random forest  regression to the  dataset 
from sklearn.ensembel import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x,y)
#visualizing the linear regression
'''plt.scatter(x,y,color='red')
plt.plot(x,linear_reg.predict(x),color='blue')
plt.title('truth or bluf (linear regression)')
plt.xlabel('position value')
plt.ylabel('salary')
plt.show()
'''
#predicting the result
y_pred=regressor.predict(6.5)

#visualizing the polynomial regression
plt.scatter(x,y,color='red')
plt.plot(x,linear_reg_2.predict(pol_reg.fit_transform(x)),color='blue')
plt.title('truth or bluf (polynomial regression)')
plt.xlabel('position value')
plt.ylabel('salary')
plt.show()

#predicting a new result with linear regression
#predicting a new result with polynomial regression







