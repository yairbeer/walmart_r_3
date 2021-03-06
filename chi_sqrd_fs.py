from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.feature_selection import chi2
from sklearn.ensemble import GradientBoostingClassifier
import xgboostlib.xgboost as xgboost


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

del train_arr

best_metric = 10
best_params = []
param_grid = {'n_estimators': [40], 'max_depth': [4, 5, 6, 7], 'max_features': [0.6],
              'learning_rate': [0.075], 'chi2_lim': [1000]}

for params in ParameterGrid(param_grid):
    print params

    # filtering low chi2 cols
    chi2_lim = params['chi2_lim']
    chi2_cols = []
    for i in range(train.shape[1]):
        if chi2_params[0][i] > chi2_lim:
            chi2_cols.append(col_list[i])

    print len(chi2_cols), ' chi2 columns'
    train_arr = train.copy(deep=True)
    train_arr = train_arr[chi2_cols]

    # Standardizing
    stding = StandardScaler()
    train_arr = stding.fit_transform(train_arr)
    classifier = GradientBoostingClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                            max_features=params['max_features'], learning_rate=params['learning_rate'])

    print 'start CV'

    # CV
    cv_n = 2
    kf = StratifiedKFold(train_result, n_folds=cv_n, shuffle=True)

    metric = []
    for train_index, test_index in kf:
        X_train, X_test = train_arr[train_index, :], train_arr[test_index, :]
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

# {'max_features': 0.8, 'n_estimators': 25, 'learning_rate': 0.1, 'max_depth': 5, 'chi2_lim': 10000}
# 93  chi2 columns
# start CV
# The log loss is:  1.03566846139
# The best metric is:  1.03566846139 for the params:  {'max_features': 0.8, 'n_estimators': 25, 'learning_rate': 0.1, 'max_depth': 5, 'chi2_lim': 10000}
# {'max_features': 0.8, 'n_estimators': 25, 'learning_rate': 0.1, 'max_depth': 5, 'chi2_lim': 5000}
# 155  chi2 columns
# start CV
# The log loss is:  1.0107879068
# The best metric is:  1.0107879068 for the params:  {'max_features': 0.8, 'n_estimators': 25, 'learning_rate': 0.1, 'max_depth': 5, 'chi2_lim': 5000}