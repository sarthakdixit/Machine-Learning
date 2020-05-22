# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 19:57:13 2020

@author: Sarthak
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt

data = pd.read_csv('headbrain.csv')
data = data.drop('Gender', axis=1)
data = data.drop('Age Range', axis=1)

train, test = train_test_split(data, test_size=0.3)
x_train = train.drop('Brain Weight(grams)', axis=1)
y_train = train['Brain Weight(grams)']
x_test = test.drop('Brain Weight(grams)', axis=1)
y_test = test['Brain Weight(grams)']

scaler = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)

rmse_val = [] 
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train, y_train)  #fit the model
    pred=model.predict(x_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
    
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()
