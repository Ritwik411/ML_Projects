# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:09:26 2019

@author: Ritwik Gupta
"""

"""
Q. (Create a program that fulfills the following specification.)
iq_size.csv

Are a person's brain size and body size (Height and weight) predictive of his or her intelligence?
Import the iq_size.csv file
It Contains the details of 38 students, where
Column 1: The intelligence (PIQ) of students
Column 2:  The brain size (MRI) of students (given as count/10,000).
Column 3: The height (Height) of students (inches)
Column 4: The weight (Weight) of student (pounds)
    What is the IQ of an individual with a given brain size of 90, height of 70 inches, and weight 150 pounds ? 
    Build an optimal model and conclude which is more useful in predicting intelligence Height, Weight or brain size.
"""
def printx():
    x=np.array([90,70,150])
    return x.reshape(1,-1)
    
def linear(features,labels):
    
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(features,labels)
    print(regressor.predict(printx()))

def polynomial(features,labels):
    from sklearn.preprocessing import PolynomialFeatures
    poly_obj = PolynomialFeatures(degree = 5)
    feature_poly = poly_obj.fit_transform(features)
    regressor = LinearRegression()
    regressor.fit(feature_poly,labels)
    print(regressor.predict(poly_obj.transform(printx())))
    

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset = pd.read_csv("iq_size.csv")

features = dataset.iloc[:,1:].values
labels = dataset.iloc[:,0].values.reshape(38,1)
linear(features,labels)
#polynomial(features,labels)

import statsmodels.api as sm
features=sm.add_constant(features)

features_opt=features[:,[0,1,2,3]]
regressor_ols=sm.OLS(endog=labels,exog=features_opt).fit()
regressor_ols.summary()

features_opt=features[:,[0,1,2]]
regressor_ols=sm.OLS(endog=labels,exog=features_opt).fit()
regressor_ols.summary()

print("Thus, it is concluded that Height and Weight are significant")

