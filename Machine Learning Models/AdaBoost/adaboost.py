# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:30:28 2020

@author: Sarthak
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data[["Outcome"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
shallow_tree = DecisionTreeClassifier(max_depth=2, random_state=0)
shallow_tree.fit(X_train, y_train)
y_pred = shallow_tree.predict(X_test)
score = metrics.accuracy_score(y_test, y_pred)
print(score)

estimators = list(range(1, 50, 5))
abc_scores = []
for n in estimators:
    abc = AdaBoostClassifier(
            base_estimator=shallow_tree,
            n_estimators=n)
    abc.fit(X_train, y_train)
    y_pred = abc.predict(X_test)
    abc_scores.append(metrics.accuracy_score(y_test, y_pred))
print(abc_scores)