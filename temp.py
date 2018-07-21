# -*- coding: utf-8 -*-
'''
#praveen kumar
#Indian institute of information technology kalyani
Data preprocessing

'''
#importing library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing dataset
dataset=pd.read_csv('C:/Users/praveen/Desktop/machine-learning udemy a-z/Data_Preprocessing/Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, :3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0 )
imputer=imputer.fit(x[:, 1:3])
x[:,1:3]=imputer.transform(x[:, 1:3])
x=pd.DataFrame(x)

#Encoding categorical data 
from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
x[:, 0]=labelencoder_x.fit_transform(x[:, 0])

#split data into two set as training set and test set
from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test=train_test_split(x,y,train_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)
 
