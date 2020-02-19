# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:04:17 2019

@author: Ritwik Gupta
"""

"""
Code Challenge 01: (Prostate Dataset)
Load the dataset from given link: 
pd.read_csv("http://www.stat.cmu.edu/~ryantibs/statcomp/data/pros.dat")

This is the Prostate Cancer dataset. Perform the train test split before you apply the model.

(a) Can we predict lpsa from the other variables?
(1) Train the unregularized model (linear regressor) and calculate the mean squared error.
(2) Apply a regularized model now - Ridge regression and lasso as well and check the mean squared error.

(b) Can we predict whether lpsa is high or low, from other variables?
"""

#Import libraries
import numpy as np 
import pandas as pd 
dataset = pd.read_csv("http://www.stat.cmu.edu/~ryantibs/statcomp/data/pros.dat",sep='\s+')

#define features and labels
features = dataset.drop('lpsa',axis=1)
labels = dataset['lpsa']

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train,labels_test	= train_test_split(features, labels, test_size=0.25, random_state=0)

from sklearn.linear_model import LinearRegression 
lm = LinearRegression()
lm.fit(features_train, labels_train)
labels_pred = lm.predict(features_test)

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(labels_test, labels_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(labels_test, labels_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels_test, labels_pred))) 

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
lm_lasso = Lasso() 
lm_ridge =  Ridge() 
lm_elastic = ElasticNet()
lm_lasso.fit(features_train, labels_train)
lm_ridge.fit(features_train, labels_train)
lm_elastic.fit(features_train, labels_train) 
predict_test_lasso = lm_lasso.predict(features_test)
predict_test_ridge = lm_ridge.predict(features_test)
predict_test_elastic = lm_elastic.predict(features_test)

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(labels_test, predict_test_lasso))  
print('Mean Squared Error:', metrics.mean_squared_error(labels_test, predict_test_lasso))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels_test, predict_test_lasso))) 

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(labels_test,predict_test_ridge))  
print('Mean Squared Error:', metrics.mean_squared_error(labels_test, predict_test_ridge))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels_test,predict_test_ridge))) 

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(labels_test,predict_test_elastic))  
print('Mean Squared Error:', metrics.mean_squared_error(labels_test,predict_test_elastic))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels_test,predict_test_elastic))) 

thresh = np.mean(labels)
labels= np.array(list(map(lambda x: 1  if x>=thresh else 0,labels)))
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train,labels_test	=	train_test_split(features, labels, test_size=0.3, random_state=0)
from sklearn.linear_model import LinearRegression 
lm = LinearRegression()
lm.fit(features_train, labels_train)
labels_pred = lm.predict(features_test)

