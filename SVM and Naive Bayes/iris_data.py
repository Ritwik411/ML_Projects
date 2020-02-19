# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 11:47:28 2019

@author: Ritwik Gupta
"""
"""

Q2. This famous classification dataset first time used in Fisher’s classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems. Iris dataset is having 4 features of iris flower and one target class.
The 4 features are
SepalLengthCm
SepalWidthCm
PetalLengthCm
PetalWidthCm
The target class
The flower species type is the target class and it having 3 types
Setosa
Versicolor
Virginica
The idea of implementing svm classifier in Python is to use the iris features to train an svm classifier and use the trained svm model to predict the Iris species type. To begin with let’s try to load the Iris dataset.
"""

import pandas as pd
import numpy as np
from sklearn import datasets

data = datasets.load_iris()
x =  data.data
y = data.target

from sklearn.model_selection import train_test_split
features_train, features_test,labels_train,labels_test = train_test_split(x,y, test_size=0.5, random_state=0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear',random_state = 0)
classifier.fit(features_train,labels_train)
pred_rbf = classifier.predict(features_test)

classifier.score(features_test,labels_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,pred_rbf)