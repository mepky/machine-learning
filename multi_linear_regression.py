# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:31:40 2018

@author: praveen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('C:/Users/praveen/Desktop/machine-learning udemy a-z/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/Multiple_Linear_Regression/50_Startups.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values
#Encoding the categorical data
#Encoding the independent variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
x[:,3]=labelencoder.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

#Avoiding the dummy varible trap
x = x[:, 1:]



#spliting the data set into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#fitting linear regression in the training set data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#predicting the test set result 
y_pred=regressor.predict(x_test)

#Building the optimal model using  backward elimination
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
x_opt=x[:, [0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
x_opt=x[:, [0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x_opt=x[:, [0,1,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()