# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:57:37 2019

@author: Ritwik Gupta
"""

"""
Code Challenge: Simple Linear Regression
  Name: 
    Food Truck Profit Prediction Tool
  Filename: 
    Foodtruck.py
  Dataset:
    Foodtruck.csv
  Problem Statement:
    Suppose you are the CEO of a restaurant franchise and are considering 
    different cities for opening a new outlet. 
    
    The chain already has food-trucks in various cities and you have data for profits 
    and populations from the cities. 
    
    You would like to use this data to help you select which city to expand to next.
    
    Perform Simple Linear regression to predict the profit based on the 
    population observed and visualize the result.
    
    Based on the above trained results, what will be your estimated profit, 
    
    If you set up your outlet in Jaipur? 
    (Current population in Jaipur is 3.073 million)
        
  Hint: 
    You will implement linear regression to predict the profits for a 
    food chain company.
    Foodtruck.csv contains the dataset for our linear regression problem. 
    The first column is the population of a city and the second column is the 
    profit of a food truck in that city. 
    A negative value for profit indicates a loss.
"""
#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Read Dataset
dataset=pd.read_csv("Foodtruck.csv")
#Allocate Feature and Label
feature = dataset.iloc[:,:-1].values
label = dataset.iloc[:,1].values
#Create an object and fit the feature and label
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(feature,label)
#regressor.coef_
#regressor.intercept_
#Prediction of data
pred=regressor.predict(np.array(3.073).reshape(1,-1))
print(pred)
if pred<0:
    print("An outlet in Jaipur would result in a loss")
else:
    print("An outlet in Jaipur would result in a Profit")
#Graphical representation
dataset.plot(x='Population',y='Profit',style='o')
plt.title('Population vs Profit')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()




