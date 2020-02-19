# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:11:04 2019

@author: Ritwik Gupta
"""

"""
Q3. 
Code Challenge -
Data: "data.csv"

This data is provided by The Metropolitan Museum of Art Open Access
1. Visualize the various countries from where the artworks are coming.
2. Visualize the top 2 classification for the artworks
3. Visualize the artist interested in the artworks
4. Visualize the top 2 culture for the artworks
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data.csv')

#1. Visualize the various countries from where the artworks are coming.
countries = dataset['Country'].dropna().value_counts()
plt.pie(countries,labels = countries.index,autopct='%1.0f%%')

#2. Visualize the top 2 classification for the artworks
classification = dataset['Classification'].dropna().value_counts().head(2)
plt.pie(classification,labels = classification.index)

#3. Visualize the artist interested in the artworks
artists = dataset['Artist Display Name'].dropna().value_counts().head(15)
plt.pie(artists,labels = artists.index)

#4. Visualize the top 2 culture for the artworks
culture = dataset['Culture'].dropna().value_counts().head(2)
plt.pie(culture,labels = culture.index)
