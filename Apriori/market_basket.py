# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:16:06 2019

@author: Ritwik Gupta
"""

"""
Code Challenge:
Datset: Market_Basket_Optimization.csv
Q2. In today's demo sesssion, we did not handle the null values before fitting the data to model, remove the null values from each row and perform the associations once again.
Also draw the bar chart of top 10 edibles.
""" 

import pandas as pd
from apyori import apriori
import matplotlib.pyplot as plt
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
dataset = dataset.fillna('null')
transactions = []
l1=[]
for i in range(0, 7501):
    temp = []
    for j in range(0, 20):    
        if str(dataset.values[i,j]) != 'null':
            temp.append(str(dataset.values[i,j]))
            l1.append(str(dataset.values[i,j]))
    transactions.append(temp)


# Training Apriori on the dataset

rules = apriori(transactions, min_support = 0.003, min_confidence = 0.25, min_lift = 4)

# Visualising the results
results = list(rules)




for item in results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
df = pd.DataFrame(l1)[0].value_counts().head(10)
plt.bar(df.index,df)
