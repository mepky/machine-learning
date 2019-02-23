# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 06:22:43 2019

@author: praveen kumar
Indian institute of information technoligy kalyani west bengal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
#from sklearn.tree import model_selection
train=pd.read_csv('train.csv')

test=pd.read_csv('test.csv')

x=train.iloc[:,1:]
y=train.iloc[:,0]
#x_test=train.iloc[:28000,0]
classifier=DecisionTreeClassifier('entropy',random_state=0)
clf=classifier.fit(x,y)

prediction=classifier.predict(test)

from sklearn.metrics import accuracy_score
scores = cross_val_score(clf,x,y, scoring = 'accuracy', cv = 10)
print(scores)
print(scores.mean())

submission=pd.DataFrame()
submission['ImageId']=test.index[1:]
submission['Label']=prediction
submission.to_csv('submissionff.csv',index=False)
