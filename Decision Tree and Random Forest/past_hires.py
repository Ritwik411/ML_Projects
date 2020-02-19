# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:43:19 2019

@author: Ritwik Gupta
"""

"""
Q1. (Create a program that fulfills the following specification.)
PastHires.csv
Here, we are building a decision tree to check if a person is hired or not based on certain predictors.
Import PastHires.csv File.
scikit-learn needs everything to be numerical for decision trees to work.
So, use any technique to map Y,N to 1,0 and levels of education to some scale of 0-2.
    Build and perform Decision tree based on the predictors and see how accurate your prediction is for a being hired.
Now use a random forest of 10 decision trees to predict employment of specific candidate profiles:
    Predict employment of a currently employed 10-year veteran, previous employers 4, went to top-tire school, having Bachelor's Degree without Internship.
    Predict employment of an unemployed 10-year veteran, ,previous employers 4, didn't went to any top-tire school, having Master's Degree with Internship.
"""
import pandas as pd  
import numpy as np  

dataset = pd.read_csv("PastHires.csv")
features = dataset.drop('Hired',axis=1)
labels = dataset['Hired']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

features['Employed?']=le.fit_transform(features['Employed?'])
features['Level of Education'] = le.fit_transform(features['Level of Education'])
features['Top-tier school'] = le.fit_transform(features['Top-tier school'])
features['Interned'] = le.fit_transform(features['Interned'])

labels = np.array(le.fit_transform(labels))
features = np.array(features)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(features,labels)
dt_pred = dt.predict(features)

from sklearn.metrics import confusion_matrix
cm_dt = confusion_matrix(labels,dt_pred)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10,random_state=0)
rf.fit(features,labels)
rf_pred = rf.predict(features)

from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(labels,rf_pred)

x = np.array([10,1,4,1,0,0]).reshape(1,-1)
y = np.array([10,0,4,1,0,1]).reshape(1,-1)

rf_predx = rf.predict(x)
rf_predy = rf.predict(y)
dt_predx = dt.predict(x)
dt_predy = dt.predict(y)
