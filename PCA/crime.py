# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:16:58 2019

@author: Ritwik Gupta
"""
"""
Q1. (Create a program that fulfills the following specification.)
Import Crime.csv File.
    Perform dimension reduction and group the cities using k-means based on Rape, Murder and assault predictors.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('crime_data.csv')
features = dataset.iloc[:,[1,2,4]].values

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#features = sc.fit_transform(features)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
features = pca.fit_transform(features)
exp_var = pca.explained_variance_ratio_

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
pred_cluster = kmeans.fit_predict(features)

plt.scatter(features[pred_cluster == 0,0], features[pred_cluster == 0,1], c = 'blue', label = 'Low')
plt.scatter(features[pred_cluster == 2,0], features[pred_cluster == 2,1], c = 'red', label = 'Medium')
plt.scatter(features[pred_cluster == 1,0], features[pred_cluster == 1,1], c = 'green', label = 'High')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], c = 'yellow', label = 'Centroids')
plt.title('Clusters of datapoints')
plt.xlabel('P1')
plt.ylabel('P2')
plt.legend()
plt.show()
