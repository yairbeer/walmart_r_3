from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
__author__ = 'WiBeer'

"""
data ML
"""
train = pd.DataFrame.from_csv("train_dummied_001_sep_b_r.csv")
train_result = np.array(pd.DataFrame.from_csv("train_result.csv")).ravel()
test = pd.DataFrame.from_csv("test_dummied_001_sep_b_r.csv")
train = np.array(train)
test = np.array(test)

# Common preprocessing
# Standardizing
stding = StandardScaler()
train = stding.fit_transform(train)
test = stding.transform(test)

# # PCA
# pcaing = PCA(n_components=100)
# train = pcaing.fit_transform(train)
# test = pcaing.transform(test)

print 'start CV'
best_metric = 10
best_params = []
param_grid = {'n_estimators': [100], 'max_features': [0.05, 0.1, 0.2], 'max_depth': [24, 32, 40],
              'min_samples_split': [5, 7, 9], 'min_samples_leaf': [1, 3]}
for params in ParameterGrid(param_grid):
    print params
    classifier = RandomForestClassifier(n_estimators=params['n_estimators'], max_features=params['max_features'],
                                        max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],
                                        min_samples_leaf=params['min_samples_leaf'])

    # CV
    cv_n = 2
    kf = StratifiedKFold(train_result, n_folds=cv_n, shuffle=True)

    metric = []
    for train_index, test_index in kf:
        X_train, X_test = train[train_index, :], train[test_index, :]
        y_train, y_test = train_result[train_index].ravel(), train_result[test_index].ravel()
        # train machine learning
        classifier.fit(X_train, y_train)

        # predict
        class_pred = classifier.predict_proba(X_test)

        # evaluate
        # print log_loss(y_test, class_pred)
        metric.append(log_loss(y_test, class_pred))

    print 'The log loss is: ', np.mean(metric)
    if np.mean(metric) < best_metric:
        best_metric = np.mean(metric)
        best_params = params
    print 'The best metric is: ', best_metric, 'for the params: ', best_params
