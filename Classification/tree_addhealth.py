# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:05:28 2019

@author: Ritwik Gupta
"""
"""
tree_addhealth.csv
Q1. (Create a program that fulfills the following specification.)
For this Code Challenge, The National Longitudinal Study of Adolescent to Adult Health (Add Health) data set, an ongoing (longitudinal) survey study that began in the mid-1990s is used. The project website URL is:
http://www.cpc.unc.edu/projects/addhealth/.
This large data set is available online from the University of North Carolina’s Carolina Population Center, http://www.cpc.unc.edu/projects/addhealth/data.
Import tree_addhealth.csv
The attributes are:
BIO_SEX: 1 = male 0 = female    
HISPANIC: 1=Yes,0=No    
WHITE : 1=Yes,0=No
BLACK : 1=Yes,0=No          
NAMERICAN: 1=Yes,0=No                      
ASIAN: 1=Yes,0=No                      
ALCEVR1: ever drank alcohol(1=Yes,0=No)   
marever1: ever smoked marijuana(1=Yes,0=No)    
cocever1: ever used cocaine(1=Yes,0=No)                
inhever1: ever used inhalants(1=Yes,0=No)             
cigavail: cigarettes available in home(1=Yes,0=No)
PASSIST: parents or public assistance(1=Yes,0=No)
EXPEL1: ever expelled from school(1=Yes,0=No)
TREG1: Ever smoked regularly(1=Yes,0=No)
Explanatory Variables:
Age
ALCPROBS1:alcohol problems 0-6
DEP1: depression scale
ESTEEM1: self esteem scale       
VIOL1:violent behaviour scale
DEVIANT1: deviant behaviour scale     
SCHCONN1: school connectedness scale       
GPA1: gpa scale  4 points)
FAMCONCT: family connectedness scale       
PARACTV:parent activities scale
PARPRES:parental presence scale
    Build a classification tree model evaluating if an adolescent would smoke regularly or not based on: gender, age, (race/ethnicity) Hispanic, White, Black, Native American and Asian, alcohol use, alcohol problems, marijuana use, cocaine use, inhalant use, availability of cigarettes in the home, depression, and self-esteem.
    Build a classification tree model evaluation if an adolescent gets expelled or not from school based on their Gender and violent behavior.
    Use random forest in relation to regular smokers as a target and explanatory variable specifically with Hispanic, White, Black, Native American and Asian.
(Please make confusion matrix and also check accuracy score for each and every section)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("tree_addhealth.csv")
dataset['age'] = dataset['age'] .fillna(np.mean(dataset['age']))
dataset['ESTEEM1'] = dataset['ESTEEM1'] .fillna(np.mean(dataset['ESTEEM1']))

#Build a classification tree model evaluating if an adolescent would smoke regularly or not based on: gender, age, (race/ethnicity) Hispanic, White, Black, Native American and Asian, alcohol use, alcohol problems, marijuana use, cocaine use, inhalant use, availability of cigarettes in the home, depression, and self-esteem.
features =  dataset.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15]].values
labels = dataset['TREG1']

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(features_train,labels_train)
dt_pred = dt.predict(features_test)

dt.score(features_test,labels_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,dt_pred)

#Build a classification tree model evaluation if an adolescent gets expelled or not from school based on their Gender and violent behavior.
features_exp = dataset[['BIO_SEX','VIOL1']].dropna()
labels_exp = dataset['EXPEL1'].dropna()

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features_exp,labels_exp,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10,random_state = 0)
rf.fit(features_train,labels_train)
pred_rf = rf.predict(features_test)

rf.score(features_test,labels_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,pred_rf)
