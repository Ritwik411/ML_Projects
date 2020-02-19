# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.cluster import DBSCAN
dbscan = DBSCAN()

print dbscan
"""
DBSCAN(eps=0.5, metric='euclidean', min_samples=5,
  random_state=111)
"""

dbscan.fit(iris.data)

dbscan.labels_

# Visualising the clusters 
# as data is in 3d space, we need to apply PCA for 2d ploting
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(iris.data)
pca_2d = pca.transform(iris.data)
explained_variance = pca.explained_variance_ratio_

df_pca =  pd.DataFrame(pca.components_,columns=iris.feature_names,index = ['PC-1','PC-2'])


"""
#alternative way, fit_transform
pca = PCA(n_components = 2)
pca_2d =  pca.fit_transform(iris.data)
"""
for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r', marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',
                        marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',
                        marker='*')
    
    
plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Noise'])
plt.title('DBSCAN finds 2 clusters and noise')
plt.savefig("dbscan.jpg")
plt.show()

"""
plt.scatter(pca_2d[dbscan.labels_ == 0,0], pca_2d[dbscan.labels_ == 0,1],c='r', marker='+' )
plt.scatter(pca_2d[dbscan.labels_ == 1,0], pca_2d[dbscan.labels_ == 1,1],c='g', marker='o' )
plt.scatter(pca_2d[dbscan.labels_ == -1,0], pca_2d[dbscan.labels_ == -1,1],c='b', marker='*' )
"""

pca.components_

for i in range(0, pca_2d.shape[0]):
    if iris.target[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r', marker='+')
    elif iris.target[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',
                        marker='o')
    elif iris.target[i] == 2:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',
                        marker='*')
    
plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Noise'])
plt.title('DBSCAN finds 2 clusters and noise')

plt.savefig("classifier.jpg")
plt.show()


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(iris.data)
for i in range(0, pca_2d.shape[0]):
    if y_kmeans[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r', marker='+')
    elif y_kmeans[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',
                        marker='o')
    elif y_kmeans[i] == 2:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',
                        marker='*')
    
plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Noise'])
plt.title('DBSCAN finds 2 clusters and noise')
plt.savefig("kmeans.jpg")

plt.show()

