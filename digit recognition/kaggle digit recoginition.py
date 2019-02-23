# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 19:47:53 2019

@author: praveen
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


sns.set(style='white', context='notebook', palette='deep')
train=pd.read_csv('C:\\Users\\praveen\\Desktop\\digit recognition\\train.csv')
test=pd.read_csv('C:\\Users\\praveen\\Desktop\\digit recognition\\test.csv')

y_train=train['label']
#print(y_train)
x_train=train.drop(labels=['label'],axis=1)

del train
g = sns.countplot(y_train)
y_train.value_counts()
x_train.isnull().any().describe()
test.isnull().any().describe()

x_train=x_train/255.0
test=test/255.0

x_train= x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

y_train = to_categorical(y_train, num_classes = 10)

random_seed=2
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1,random_state=random_seed)

g = plt.imshow(x_train[0][:,:,0])


