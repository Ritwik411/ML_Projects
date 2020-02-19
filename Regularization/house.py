# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:45:40 2019

@author: Ritwik Gupta
"""
"""
Code Challenges 02: (House Data)
This is kings house society data.
In particular, we will: 
• Use Linear Regression and see the results
• Use Lasso (L1) and see the resuls
• Use Ridge and see the score
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("kc_house_data.csv")
dataset['sqft_above'] = dataset['sqft_above'].fillna(np.mean(dataset['sqft_above']))
features = dataset.drop(['id','date','price','zipcode','lat','long'],axis=1)
labels = dataset.iloc[:,2].values
features = np.array(features)


from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2,random_state=0)



from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
lm = LinearRegression()
llm = Lasso()
lrm = Ridge()
lem = ElasticNet()
lm.fit(features_train,labels_train)
llm.fit(features_train,labels_train)
lrm.fit(features_train,labels_train)
lem.fit(features_train,labels_train)

print ("RSquare Value for Simple Regresssion TEST data is-") 
print (np.round (lm.score(features_test,labels_test)*100,2))

print ("RSquare Value for Lasso Regresssion TEST data is-")
print (np.round (llm.score(features_test,labels_test)*100,2))

print ("RSquare Value for Ridge Regresssion TEST data is-")
print (np.round (lrm.score(features_test,labels_test)*100,2))

print ("RSquare Value for Elastic Net Regresssion TEST data is-")
print (np.round (lem.score(features_test,labels_test)*100,2))

