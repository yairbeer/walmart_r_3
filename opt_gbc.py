from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
__author__ = 'WiBeer'

"""
data ML
"""
train = pd.DataFrame.from_csv("train_dummied_200_sep_dep_fln_b_r_v2.csv")
train_result = np.array(pd.DataFrame.from_csv("train_result.csv")).ravel()
train = np.array(train)

print train.shape[1], ' columns'
# Common preprocessing
# Standardizing
stding = StandardScaler()
train = stding.fit_transform(train)

print 'start CV'
best_metric = 10
best_params = []
param_grid = {'n_estimators': [25], 'max_depth': [10], 'max_features': [0.2]}

for params in ParameterGrid(param_grid):
    print params
    classifier = GradientBoostingClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                            max_features=params['max_features'])

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
