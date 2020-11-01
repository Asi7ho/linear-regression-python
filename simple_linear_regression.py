#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 19:55:14 2019

@author: yanndebain
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
dataSet = pd.read_csv('/Users/yanndebain/MEGA/MEGAsync/Code/Data Science/ML/Linear Regression/Salary_Data.csv')

X = dataSet.iloc[:, :-1].values #independant variables
y = dataSet.iloc[:, -1].values #dependant variables

# Training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1.0/3)

# Building model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Test model
y_pred = regressor.predict(X_test)
#regressor.predict([[15]])


# Data visualization
plt.scatter(X_test, y_test, color = 'red')
#plt.scatter(X_train, y_train, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
