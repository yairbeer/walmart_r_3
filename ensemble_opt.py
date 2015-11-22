from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
import xgboostlib.xgboost as xgboost
import glob

__author__ = 'YBeer'

train_result = pd.DataFrame.from_csv("train_result.csv")
col = list(train_result.columns.values)
result_ind = list(train_result[col[0]].value_counts().index)
# train_result = pd.get_dummies(train_result)
train_result = np.array(train_result).ravel()
train_result_xgb = np.zeros(train_result.shape)
for i in range(1, len(result_ind)):
    train_result_xgb += (train_result == result_ind[i]) * i
# print train_result_xgb

# combining meta_estimators
train = glob.glob('meta*')
print train
for i in range(len(train)):
    train[i] = pd.DataFrame.from_csv(train[i])
train = pd.concat(train, axis=1)

train_arr = np.array(train)
col_list = list(train.columns.values)

# print train_result.shape[1], ' categorial'
print train.shape[1], ' columns'

best_metric = 10
best_params = []
param_grid = {'silent': [1], 'nthread': [2], 'num_class': [38], 'eval_metric': ['mlogloss'], 'eta': [0.1],
              'objective': ['multi:softprob'], 'max_depth': [4, 5, 6], 'chi2_lim': [0], 'num_round': [300],
              'subsample': [0.5, 0.75, 1]}

for params in ParameterGrid(param_grid):
    print params

    # Standardizing
    stding = StandardScaler()
    train_arr = stding.fit_transform(train_arr)

    print 'start CV'
    # CV
    cv_n = 4
    kf = StratifiedKFold(train_result, n_folds=cv_n, shuffle=True)
    metric = []
    meta_estimator_xgboost = np.zeros((train_arr.shape[0], 38))
    for train_index, test_index in kf:
        X_train, X_test = train_arr[train_index, :], train_arr[test_index, :]
        y_train, y_test = train_result_xgb[train_index].ravel(), train_result_xgb[test_index].ravel()

        # train machine learning
        xg_train = xgboost.DMatrix(X_train, label=y_train)
        xg_test = xgboost.DMatrix(X_test, label=y_test)

        watchlist = [(xg_train, 'train'), (xg_test, 'test')]

        num_round = params['num_round']
        xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);

        # predict
        class_pred = xgclassifier.predict(xg_test)
        class_pred = class_pred.reshape(y_test.shape[0], 38)

        meta_estimator_xgboost[test_index, :] = class_pred

        # evaluate
        # print log_loss(y_test, class_pred)
        metric = log_loss(y_test, class_pred)

    print 'The log loss is: ', metric
    if metric < best_metric:
        best_metric = metric
        best_params = params
    print 'The best metric is:', best_metric, 'for the params:', best_params
