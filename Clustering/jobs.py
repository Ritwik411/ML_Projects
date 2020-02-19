# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:55:55 2019

@author: Ritwik Gupta
"""

"""
Code Challenge - 
 This is a pre-crawled dataset, taken as subset of a bigger dataset 
 (more than 4.7 million job listings) that was created by extracting data 
 from Monster.com, a leading job board.
 Remove location from Organization column?
 Remove organization from Location column?
 In Location column, instead of city name, zip code is given, deal with it?
 Seperate the salary column on hourly and yearly basis and after modification
 salary should not be in range form , handle the ranges with their average
 Which organization has highest, lowest, and average salary?
 which Sector has how many jobs?
 Which organization has how many jobs
 Which Location has how many jobs?
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

# Importing the dataset 
dataset = pd.read_csv('monster_com-job_sample.csv')

def comma_sep(str1):
    str1 = re.findall(r'\w+\,\W\w*\W?\d*\w*|\w+\W\w+\,\W\w+\W\d+|\w+\W\d+|\w+\W\w+\,\W\w+|\w+\W\w+\,\W\w*\W?\d*\w*',str1)
    return str1                

for i in range (0,len(dataset)):
    if re.match(r'[A-Za-z]+\,\W[A-Z]{2}\W[0-9]*|\w+\W\w+\W\,\W[A-Z]{2}',str(dataset['organization'][i])):
        dataset['location'][i],dataset['organization'][i] = dataset['organization'][i],dataset['location'][i]  

for i in range(0,len(dataset['location'])):
    if len(dataset['location'][i]) >35:
        dataset['location'][i] = 'nan'        
dataset["location"] = dataset["location"].apply(comma_sep)        
        