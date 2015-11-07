from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.feature_selection import chi2
from sklearn.ensemble import GradientBoostingClassifier


__author__ = 'YBeer'

train = pd.DataFrame.from_csv("train_dummied_200_sep_dep_fln_b_r_v2.csv")
train.fillna(0)
train_arr = np.array(train)
col_list = list(train.columns.values)

train_result = pd.DataFrame.from_csv("train_result.csv")
# train_result = pd.get_dummies(train_result)
train_result = np.array(train_result).ravel()

# print train_result.shape[1], ' categorial'
print train.shape[1], ' columns'

print 'absing'
for i in range(train_arr.shape[0]):
    for j in range(train_arr.shape[1]):
        train_arr[i, j] = np.abs(train_arr[i, j])

chi2_params = chi2(train_arr, train_result)

# for i in range(train_arr.shape[1]):
#     print col_list[i], chi2_params[0][i]

# filtering low chi2 cols
chi2_lim = 1000
chi2_cols = []
for i in range(train.shape[1]):
    if chi2_params[0][i] > chi2_lim:
        chi2_cols.append(col_list[i])

del train_arr

print len(chi2_cols), ' chi2 columns'
train = train[chi2_cols]

# Standardizing
stding = StandardScaler()
train = stding.fit_transform(train)

print 'start CV'
best_metric = 10
best_params = []
param_grid = {'n_estimators': [50], 'max_depth': [5], 'max_features': [0.6],
              'learning_rate': [0.03, 0.06, 0.1, 0.3, 0.6]}


for params in ParameterGrid(param_grid):
    print params
    classifier = GradientBoostingClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                            max_features=params['max_features'], learning_rate=params['learning_rate'])

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

# The best metric is:  0.968018334947 for the params:  {'max_features': 0.1, 'min_samples_split': 15, 'n_estimators': 100, 'max_depth': 60, 'min_samples_leaf': 1}
