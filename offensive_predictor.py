# -*- coding: utf-8 -*-
"""offensive-predictor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LAjjRDSBuVJU5up00V1EiLArFAFezTkC
"""

# CSE 30124 MLB Contract Evaluator
# Evan Ackerman

# Importing Libraries
import builtins
import pandas as pd
import csv
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

import random as rand

import seaborn

# Loading in the File
import os

# THIS CURRENT DATA IS QUALIFIED BATTERS FROM 2015 - 2022 (502 PA minimum)
if not os.path.exists('stats_2015_2022.csv'):
    !wget -c https://drive.google.com/file/d/1Lny1ZkLRuxtJda0CieGVLGwUduy9BXCt/view\?usp=sharing -O stats_2015_2022.csv

stats = pd.read_csv('stats_2015_2022.csv')

# display what stats we have
cols = []
for col in stats.columns:
    cols.append(col)
print(cols)

# adjust the salaries that are in our data set to accound for inflation
INFLATION_CONVERSION = {
                            2015: 1.299,
                            2016: 1.282,
                            2017: 1.256,
                            2018: 1.226,
                            2019: 1.204,
                            2020: 1.189,
                            2021: 1.136,
                            2022: 1.052
                        }
million = 1000000
nSalaries = len(stats[' salary '])

salaries = list()
years = list()
for sal in stats[' salary ']:
    salaries.append(sal)

for y in stats['year']:
    years.append(y)

# Salaries Before Inflation
print(f'Average Salary before Inflation:')
avg = sum(salaries)/len(salaries)
print(f'${avg/million:.2f} Million')

for i in range(nSalaries):
    if years[i] in INFLATION_CONVERSION.keys():
        salaries[i] *= INFLATION_CONVERSION[years[i]]

# Salaries After Inflation
print(f'\nAverage Salary after Inflation:')
final_avg = sum(salaries)/len(salaries)
print(f'${final_avg/million:.2f} Million')

# Analysis of Stats Together
seaborn.pairplot(stats)

# Wow that's a large image, let's focus in!
pal = seaborn.color_palette("bright", 22)
seaborn.relplot(data=stats, x='player_age', y='WAR', hue='player_age', palette=pal)

from sklearn import linear_model
# Break up our data

# Getting our player ages and WARS
X = stats['player_age']
X = np.array(X).reshape(-1,1)
Y = stats['WAR']

# Now that we have the Ages
# TRAINING 75% TESTING 25%



#model = linear_model.BayesianRidge()
poly = PolynomialFeatures()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_test = np.sort(X_test, axis=0)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Train the Model
WAR_model = LinearRegression()
WAR_model.fit(X_poly_train,Y_train)

Y_pred = WAR_model.predict(X_poly_train)
Y_predTest = WAR_model.predict(X_poly_test)

plt.xlabel("Player Age")
plt.ylabel("WAR")
plt.grid()
plt.title("MLB Qualified Hitter WAR vs Age")
plt.scatter(X, Y, color='b', label ="Actual Player Data")
plt.plot(X_test, Y_predTest, color="r")
plt.axis([20,40,-2,12])
plt.legend("Actual Data", "Predicted Trend")
plt.show()