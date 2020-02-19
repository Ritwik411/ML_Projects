# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 10:44:43 2019

@author: Ritwik Gupta
"""

"""
Q1. Human Activity Recognition
Human Activity Recognition with Smartphones
(Recordings of 30 study participants performing activities of daily living)
(Click Here To Download Dataset): https://github.com/K-Vaid/Python-Codes/blob/master/Human_activity_recog.zip
In an experiment with a group of 30 volunteers within an age bracket of 19 to 48 years, each person performed six activities (WALKING, WALKING UPSTAIRS, WALKING DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. The experiments have been video-recorded to label the data manually.
The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data.
Attribute information 
For each record in the dataset the following is provided:
        Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration. 
        Triaxial Angular velocity from the gyroscope.
        A 561-feature vector with time and frequency domain variables.
        Its activity labels.
        An identifier of the subject who carried out the experiment.
Train a tree classifier to predict the labels from the test data set using the following approaches:
  (a) a decision tree approach,
  (b) a random forest approach and
  (c) a logistic regression.
  (d) KNN approach
Examine the result by reporting the accuracy rates of all approach on both the testing and training data set. Compare the results. Which approach would you recommend and why?
        Perform feature selection and repeat the previous step. Does your accuracy improve?
        Plot two graph showing accuracy bar score of all the approaches taken with and without feature selection.
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

dataset = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

features = dataset.iloc[:,:-1].values
labels = dataset.iloc[:,-1].values

features_test = test.iloc[:,:-1].values
labels_test = test.iloc[:,-1].values

#(a) a decision tree approach
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(features,labels)

pred = dt.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,pred)

s1 = dt.score(features_test,labels_test)

#(b) a random forest approach

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 50,random_state = 0)
rf.fit(features,labels)
rf_pred = rf.predict(features_test)
s2 = rf.score(features_test,labels_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(features,labels)
lr_pred = lr.predict(features_test)
s3 = lr.score(features_test,labels_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5,p = 2)
knn.fit(features,labels)
knn_pred = knn.predict(features_test)
s4 = knn.score(features_test,labels_test)

#Feature Selection
import statsmodels.api as sm
features = sm.add_constant(features)
regreessor_ols = sm.OLS(endog = labels,exog = features).fit()
regressor_ols.pvalues

plt.bar(['DT','RF','LR','KNN'],[s1,s2,s3,s4])
plt.xlabel('Approach')
plt.ylabel('Score')
plt.show()
    
