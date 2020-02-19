# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:26:01 2019

@author: Ritwik Gupta
"""

"""


Q. (Create a program that fulfills the following specification.)
Female_Stats.Csv

Female Stat Students

 

Import The Female_Stats.Csv File

The Data Are From N = 214 Females In Statistics Classes At The University Of California At Davi.

Column1 = Student’s Self-Reported Height,

Column2 = Student’s Guess At Her Mother’s Height, And

Column 3 = Student’s Guess At Her Father’s Height. All Heights Are In Inches.

 

Build A Predictive Model And Conclude If Both Predictors (Independent Variables) Are Significant For A Students’ Height Or Not
When Father’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Mother’s Height.
When Mother’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Father’s Height.

"""
#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
#Load the dataset
dataset=pd.read_csv("Female_Stats.csv")
feature=dataset.iloc[:,1:].values
#feature_dad=dataset.iloc[:,2]
label=dataset.iloc[:,0].values.reshape(214,1)
feature=sm.add_constant(feature)
#feature_dad=sm.add_constant(feature_dad)
features_opt=feature[:,:]
regressor_ols=sm.OLS(endog=label,exog=features_opt).fit()
regressor_ols.summary()
print("Heights of both parents are significant")

feature_mom = dataset.iloc[:,1:].values
init_mean=np.mean(label)
#When Father’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Mother’s Height.
dataset['feature_mom']=list(map(lambda x:x+1,feature_mom[:,0]))
feature_mom1 = dataset.iloc[:,[2,3]].values
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(feature_mom,np.array(label).reshape(-1,1))
pred_mom = regressor.predict(feature_mom1)
Final_mean2=np.mean(pred_mom)
print("Change in height=",regressor.coef_)

#When Mother’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Father’s Height.
dataset['feature_dad']=list(map(lambda x:x+1,feature_mom[:,1]))
feature_dad1 = dataset.iloc[:,[1,4]].values
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(feature_mom,np.array(label).reshape(-1,1))
pred_dad = regressor.predict(feature_dad1)
Final_mean1=np.mean(pred_dad)


