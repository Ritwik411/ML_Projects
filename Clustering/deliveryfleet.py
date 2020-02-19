# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:46:06 2019

@author: Ritwik Gupta
"""
""" 
Q1. (Create a program that fulfills the following specification.)
deliveryfleet.csv
Import deliveryfleet.csv file
Here we need Two driver features: mean distance driven per day (Distance_feature) and the mean percentage of time a driver was >5 mph over the speed limit (speeding_feature).
    Perform K-means clustering to distinguish urban drivers and rural drivers.
    Perform K-means clustering again to further distinguish speeding drivers from those who follow speed limits, in addition to the rural vs. urban division.
    Label accordingly for the 4 groups.
"""
#Importing the libraires
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Loading the dataset
dataset = pd.read_csv("deliveryfleet.csv")
features = dataset.iloc[:,[1,2]].values

plt.scatter(features[:,0],features[:,1])
plt.show()

#Perform K-means clustering to distinguish urban drivers and rural drivers.
kmeans = KMeans(n_clusters = 2,init = 'k-means++',random_state = 0)
pred = kmeans.fit_predict(features)

plt.scatter(features[pred == 0,0],features[pred == 0,1],c='green',label='Rural')
plt.scatter(features[pred == 1,0],features[pred == 1,1],c='red',label='Urban')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c= 'blue',label = 'Centroids')
plt.title('Cluster of Rural and Urban Drivers')
plt.xlabel('Mean Distance per day')
plt.ylabel('Mean % of time')
plt.legend()
plt.show()

#Perform K-means clustering again to further distinguish speeding drivers from those who follow speed limits, in addition to the rural vs. urban division.
kmeans_speed = KMeans(n_clusters = 4,init = 'k-means++',random_state = 0)
pred_speed = kmeans_speed.fit_predict(features)

plt.scatter(features[pred_speed == 0,0],features[pred_speed == 0,1],c = 'blue',label = 'Rural Low speed')
plt.scatter(features[pred_speed == 1,0],features[pred_speed == 1,1],c = 'black',label = 'Rural High speed')
plt.scatter(features[pred_speed == 2,0],features[pred_speed == 2,1],c = 'green',label = 'Urban Low speed')
plt.scatter(features[pred_speed == 3,0],features[pred_speed == 3,1],c = 'orange',label = 'Urban High speed')
plt.scatter(kmeans_speed.cluster_centers_[:,0],kmeans_speed.cluster_centers_[:,1],c= 'red',label = 'Centroids')
plt.title('Total Representation')
plt.xlabel('Mean Distance per day')
plt.ylabel('Mean % of time')

plt.show()
