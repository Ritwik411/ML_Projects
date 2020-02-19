# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:56:58 2019

@author: Ritwik Gupta
"""
"""
Q2. Code Challenge (Connecting Hearts)
Downlaod Link: http://openedx.forsk.in/c4x/Manipal_University/FL007/asset/Resource.zip
What influences love at first sight? (Or, at least, love in the first four minutes?) This dataset was compiled by Columbia Business School Professors Ray Fisman and Sheena Iyengar for their paper Gender Differences in Mate Selection: Evidence from a Speed Dating Experiment.
Data was gathered from participants in experimental speed dating events from 2002-2004. During the events, the attendees would have a four minute "first date" with every other participant of the opposite sex. At the end of their four minutes, participants were asked if they would like to see their date again.
They were also asked to rate their date on six attributes: Attractiveness, Sincerity, Intelligence, Fun, Ambition, and Shared Interests.
The dataset also includes questionnaire data gathered from participants at different points in the process.
These fields include: demographics, dating habits, self-perception across key attributes, beliefs on what others find valuable in a mate, and lifestyle information.
See the Key document attached for details of every column and for the survey details.
What does a person look for in a partner? (both male and female)
For example: being funny is more important for women than man in selecting a partner! Being sincere on the other hand is more important to men than women.
    What does a person think that their partner would look for in them? Do you think what a man thinks a woman wants from them matches to what women really wants in them or vice versa. TIP: If it doesn’t then it will be one sided :)
    Plot Graphs for:
            How often do they go out (not necessarily on dates)?
            In which activities are they interested?  
    If the partner is from the same race are they more keen to go for a date?
    What are the least desirable attributes in a male partner? Does this differ for female partners?
    How important do people think attractiveness is in potential mate selection vs. its real impact?
"""
def plot(l1,l2):
    l2= ['Attractive','Sincere','Intelligent','Fun','Ambitious','Shared']
    plt.bar(l2,l1)
    plt.xlabel('Traits')
    plt.ylabel('Value')
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
dataset = pd.read_csv('Dating_Data.csv',delimiter = ',',encoding = 'Windows 1252')
dataset_male = dataset[dataset['gender'] == 1]
dataset_female = dataset[dataset['gender'] == 0]
l1 = []
l2 = []
#Male representation
l1 = [np.mean(np.array(dataset_male['attr1_1'].dropna())).astype(int),np.mean(np.array(dataset_male['sinc1_1'].dropna()).astype(int)),np.mean(np.array(dataset_male['intel1_1'].dropna()).astype(int)),np.mean(np.array(dataset_male['fun1_1'].dropna()).astype(int)),np.mean(np.array(dataset_male['amb1_1'].dropna()).astype(int)),np.mean(np.array(dataset_male['shar1_1'].dropna()).astype(int))]
plot(l1,l2)
#Female Representation
l1 = [np.mean(np.array(dataset_female['attr1_1'].dropna())).astype(int),np.mean(np.array(dataset_female['sinc1_1'].dropna()).astype(int)),np.mean(np.array(dataset_female['intel1_1'].dropna()).astype(int)),np.mean(np.array(dataset_female['fun1_1'].dropna()).astype(int)),np.mean(np.array(dataset_female['amb1_1'].dropna()).astype(int)),np.mean(np.array(dataset_female['shar1_1'].dropna()).astype(int))]
plot(l1,l2)
#What does a person think that their partner would look for in them? Do you think what a man thinks a woman wants from them matches to what women really wants in them or vice versa. TIP: If it doesn’t then it will be one sided :)
l2= ['Attractive','Sincere','Intelligent','Fun','Ambitious']
l3 = [np.mean(np.array(dataset_male['attr2_1'].dropna())).astype(int),np.mean(np.array(dataset_male['sinc2_1'].dropna()).astype(int)),np.mean(np.array(dataset_male['intel2_1'].dropna()).astype(int)),np.mean(np.array(dataset_male['fun2_1'].dropna()).astype(int)),np.mean(np.array(dataset_male['amb2_1'].dropna()).astype(int))]
plt.bar(l2,l3)
#How often do they go out (not necessarily on dates)?
plt.pie(dataset['go_out'].dropna().drop_duplicates(keep = 'first'),labels = range(1,8),autopct='%1.1f%%')
l1 = np.mean(dataset.iloc[:,50:67].dropna()).values
#In which activities are they interested?  
sb.barplot(l1,['sports','tvsports','excersice','dining','museums','art','hiking','gaming','clubbing','reading','tv','theater','movies','concerts','music','shopping','yoga'],orient='h')
#If the partner is from the same race are they more keen to go for a date?
dataset_race = dataset[(dataset['match'] == 1) & dataset['samerace'] == 1]
print(len(dataset_race)/len(dataset))
#What are the least desirable attributes in a male partner? Does this differ for female partners?
l1 = [np.mean(np.array(dataset_female['attr1_3'].dropna())).astype(int),np.mean(np.array(dataset_female['sinc1_3'].dropna()).astype(int)),np.mean(np.array(dataset_female['intel1_3'].dropna()).astype(int)),np.mean(np.array(dataset_female['fun1_3'].dropna()).astype(int)),np.mean(np.array(dataset_female['amb1_3'].dropna()).astype(int)),np.mean(np.array(dataset_female['shar1_3'].dropna()).astype(int))]
l2= ['Attractive','Sincere','Intelligent','Fun','Ambitious','Shared']
l3 = pd.Series(l1,index = l2).sort_values()
print(l3.index[0])
#How important do people think attractiveness is in potential mate selection vs. its real impact?
dataset_pot = dataset[dataset['match'] == 1] 