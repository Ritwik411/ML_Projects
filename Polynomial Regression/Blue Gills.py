# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:30:00 2019

@author: Ritwik Gupta
"""
"""
Q. (Create a program that fulfills the following specification.)
bluegills.csv

How is the length of a bluegill fish related to its age?

In 1981, n = 78 bluegills were randomly sampled from Lake Mary in Minnesota. The researchers (Cook and Weisberg, 1999) measured and recorded the following data (Import bluegills.csv File)

Response variable(Dependent): length (in mm) of the fish

Potential Predictor (Independent Variable): age (in years) of the fish

    
    

NOTE: Observe that 80.1% of the variation in the length of bluegill fish is reduced by taking into account a quadratic function of the age of the fish.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("bluegills.csv")
feature = dataset.iloc[:,0].values.reshape(-1,1)
label = dataset.iloc[:,1].values.reshape(-1,1)
#How is the length of a bluegill fish best related to its age? (Linear/Quadratic nature?)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(feature,label)
pred1 = regressor.predict(feature)
plt.scatter(feature,label)
plt.plot(feature,pred1)

#What is the length of a randomly selected five-year-old bluegill fish? Perform polynomial regression on the dataset.
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=5)
length_poly=poly.fit_transform(feature)
regressor.fit(length_poly,label)
red3 = regressor.predict(poly.transform(feature))
pred2 = regressor.predict(poly.transform(np.array(5).reshape(1,-1)))
plt.scatter(feature,label)
plt.plot(feature,red3)
