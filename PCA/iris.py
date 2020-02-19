# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:03:13 2019

@author: Ritwik Gupta
"""
"""
Q2. (Create a program that fulfills the following specification.)
The iris data set consists of 50 samples from each of three species of Iris flower (Iris setosa, Iris virginica and Iris versicolor).
    Four features were measured from each sample: the length and the width of the sepals and petals, in centimetres (iris.data).
    Import the iris dataset already in sklearn module using the following command:
from sklearn.datasets import load_iris
iris = load_iris()
iris=iris.data
Reduce dimension from 4-d to 2-d and perform clustering to distinguish the 3 species.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
iris_main = load_iris()
iris=iris_main.data
target = iris_main.target

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3,init = 'k-means++',random_state = 0)
pred_cluster =  kmeans.fit_predict(iris)

plt.scatter(iris[pred_cluster == 0, 0], iris[pred_cluster == 0, 1], c = 'blue', label = 'Versicolor')
plt.scatter(iris[pred_cluster == 1, 0], iris[pred_cluster == 1, 1], c = 'red', label = 'Setosa')
plt.scatter(iris[pred_cluster == 2, 0], iris[pred_cluster == 2, 1], c = 'green', label = 'Virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'Centroids')
plt.title('Clusters of datapoints')
plt.xlabel('X Cordindates')
plt.ylabel('Y Cordinates')
plt.legend()
plt.show()