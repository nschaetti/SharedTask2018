#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import unicodecsv
import codecs
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# CSV reader
r = unicodecsv.reader(codecs.open('./data/cwi-train-cx-1.csv', 'rb', encoding='utf-8'))

# Samples
X = np.zeros((27299, 416))
Y = np.zeros(27299)

# Read
for index, row in enumerate(r):
    # For each feature
    for i in range(416):
        X[index, i] = float(row[i])
    # end for

    # Label
    Y[index] = int(row[-1])
# end for

# Decision Tree classifier
dt_clf = DecisionTreeClassifier(max_depth=None)

# Test
print(u"Testing decision trees")
dt_clf.fit(X, Y)
print(u"Average decision tree training accuracy : {}".format(dt_clf.score(X, Y)))
scores = cross_val_score(dt_clf, X, Y)
print(u"Average decision tree test accuracy : {}".format(scores.mean()))

# Random forest
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None)

# Test
print(u"Testing random forest")
scores = cross_val_score(rf_clf, X, Y)
print(u"Average random forest accuracy : {}".format(scores.mean()))

# Extra trees classifier
et_clf = ExtraTreesClassifier(n_estimators=100, max_depth=None)

# Test
print(u"Testing extra trees")
scores = cross_val_score(et_clf, X, Y)
print(u"Average extra trees classifier : {}".format(scores.mean()))

# AdaBoost classifier
ada_clf = AdaBoostClassifier(n_estimators=100)

# Test
print(u"Testing AdaBoost")
scores = cross_val_score(ada_clf, X, Y)
print(u"Average AdaBoost accuracy : {}".format(scores.mean()))

# Gradient Tree boosting
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0)

# Test
print(u"Testing gradient boosting")
scores = cross_val_score(gb_clf, X, Y)
print(u"Average gradient boosting accuracy : {}".format(scores.mean()))
