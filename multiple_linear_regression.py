#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 20:02:10 2019

@author: yanndebain
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
dataSet = pd.read_csv('/Users/yanndebain/MEGA/MEGAsync/Code/Data Science/ML/Linear Regression/50_Startups.csv')

X = dataSet.iloc[:, :-1].values #independant variables
y = dataSet.iloc[:, -1].values #dependant variables


# Categoric variables
columnTransformer = ColumnTransformer([('dummyColumn', OneHotEncoder(), [3])], remainder='passthrough')
X = columnTransformer.fit_transform(X)
X = X[:, 1:] # We only keep 2 out of 3 dummy columns [California, Florida, New-York] -> [Florida, New-York]


#Training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Building model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Test model
y_pred = regressor.predict(X_test)
#regressor.predict(np.array([[1, 0, 130000, 140000, 300000]]))




