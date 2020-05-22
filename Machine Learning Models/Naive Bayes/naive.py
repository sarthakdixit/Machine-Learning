# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 07:57:45 2020

@author: Sarthak
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

data = datasets.load_wine()
print(data.data)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=109)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))