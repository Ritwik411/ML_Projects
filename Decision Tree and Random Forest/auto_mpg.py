# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:29:12 2019

@author: Ritwik Gupta
"""

"""
Q1. (Create a program that fulfills the following specification.)
Auto_mpg.txt

Here is the dataset about cars. The data concerns city-cycle fuel consumption in miles per gallon (MPG).

    Import the dataset Auto_mpg.txt
    Give the column names as "mpg", "cylinders", "displacement","horsepower","weight","acceleration", "model year", "origin", "car name" respectively
    Display the Car Name with highest miles per gallon value
    Build the Decision Tree and Random Forest models and find out which of the two is more accurate in predicting the MPG value
    Find out the MPG value of a 80's model car of origin 3, weighing 2630 kgs with 6 cylinders, having acceleration around 22.2 m/s due to it's 100 horsepower engine giving it a displacement of about 215. (Give the prediction from both the models)
"""
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from statistics import mean
dataset = pd.read_csv("Auto_mpg.txt",sep='\s+')
dataset.columns = ["mpg", "cylinders", "displacement","horsepower","weight","acceleration", "model year", "origin", "car name"]
print(dataset.sort_values('mpg',ascending=False)['car name'].head(1))
l1=list(dataset['horsepower'])
for item in l1:
    if item=='?':
        l1.remove(item)
mean_rep = str(mean(list(map(float,l1))))
dataset['horsepower'] = dataset['horsepower'].str.replace('?',mean_rep)
features = dataset.drop(['car name','mpg'],axis=1)
labels = dataset.iloc[:,0].values

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)  
features_test = sc.transform(features_test)


#DecisionTree approach
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(features_train,labels_train)
pred_dt = dtr.predict(features_test)
print(dtr.score(features_test,labels_test))
print(dtr.score(features_train,labels_train))

#RandomForest approach
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=26,random_state=0)
rf.fit(features_train,labels_train)
pred_rf = rf.predict(features_test)
print(rf.score(features_test,labels_test))
print(rf.score(features_train,labels_train))

x=np.array([6,215,100,2630,22.2,80,3])
rf = RandomForestRegressor(n_estimators=26,random_state=0)
rf.fit(features_train,labels_train)
print(rf.predict(x.reshape(1,-1)))
print(dtr.predict(x.reshape(1,-1)))



