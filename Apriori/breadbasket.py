# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:13:35 2019

@author: Ritwik Gupta
"""

""" 
Code Challenge:
dataset: BreadBasket_DMS.csv

Q1. In this code challenge, you are given a dataset which has data and time wise transaction on a bakery retail store.
1. Draw the pie chart of top 15 selling items.
2. Find the associations of items where min support should be 0.0025, min_confidence=0.2, min_lift=3.
3. Out of given results sets, show only names of the associated item from given result row wise.
""" 
import pandas as pd
from apyori import apriori
import matplotlib.pyplot as plt

dataset = pd.read_csv("BreadBasket_DMS.csv")

def my_func(value):
    str1 = ",".join(value)
    return list(set(str1.split(',')))
#1. Draw the pie chart of top 15 selling items.
dataset = dataset.mask(dataset.eq('NONE')).dropna().reset_index()
dataset = dataset.iloc[:,1:]

top_15 = dataset['Item'].value_counts().head(15)
plt.pie(top_15,labels=top_15.index)

#2. Find the associations of items where min support should be 0.0025, min_confidence=0.2, min_lift=3.
df_grp = list(dataset.groupby('Transaction')['Item'].apply(my_func))

rules = apriori(df_grp, min_support = 0.0025,min_confidence = 0.2,min_lift = 3)
next(rules)
results = list(rules)

for item in results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print(items)
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
