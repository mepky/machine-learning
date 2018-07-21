# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 17:27:55 2018

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
'''from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x,y)'''

#spliting the dataset into training set into test set
'''from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
'''
#fetature scaling
'''from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x_train=sc_x.fit_transform(x_train)
y_train=sc_y.fit_transform(y_train)
x_test=sc_x.fit(x_test)
y_test=sc_y.fit(y_test)'''
#Fitting decision tree regression to the  dataset 
from sklearn.tree import DecisionTreeRegression
regressor=DecisionTreeRegression(random_state=0)
regressor.fit(x,y)

#visualizing the linear regression
'''plt.scatter(x,y,color='red')
plt.plot(x,linear_reg.predict(x),color='blue')
plt.title('truth or bluf (linear regression)')
plt.xlabel('position value')
plt.ylabel('salary')
plt.show()
'''
#predict the value
y_pred=regressor.predict(6.5)

#visualizing the decision tree regression
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('truth or bluf (polynomial regression)')
plt.xlabel('position value')
plt.ylabel('salary')
plt.show()






