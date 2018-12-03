# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 22:35:52 2018

@author: praveen kumar

    Indian institute of information technology kalyani W.B
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#data preprocessing
dataset=pd.read_csv("C:\\Users\\praveen\\Desktop\\machine-learning udemy a-z\\Machine Learning A-Z Template Folder\\Part 7 - Natural Language Processing\\Section 36 - Natural Language Processing\\Natural_Language_Processing\\Restaurant_Reviews.tsv",delimiter='\t',quoting=3)
#data cleaning
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    review=[word for word in review if not word in set(stopwords.words('english'))]
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values
