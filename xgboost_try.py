from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.feature_selection import chi2
import xgboostlib.xgboost as xgboost

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

train = pd.DataFrame.from_csv("train_dummied_200_sep_dep_fln_b_r_v5.csv")
train.fillna(0)
train_arr = np.array(train)
col_list = list(train.columns.values)

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
param_grid = {'silent': [1], 'nthread': [4], 'num_class': [38], 'eval_metric': ['mlogloss'], 'eta': [0.1],
              'objective': ['multi:softprob'], 'max_depth': [4], 'chi2_lim': [0], 'num_round': [500]}

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

    print 'start CV'

    # CV
    cv_n = 2
    X_train, X_test, y_train, y_test = train_test_split(train_arr, train_result_xgb, test_size=0.5, random_state=1)
    metric = []

    # train machine learning
    xg_train = xgboost.DMatrix(X_train, label=y_train)
    xg_test = xgboost.DMatrix(X_test, label=y_test)

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]

    num_round = params['num_round']
    xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);

    # predict
    class_pred = xgclassifier.predict(xg_test)
    class_pred = class_pred.reshape(y_test.shape[0], 38)

    # evaluate
    # print log_loss(y_test, class_pred)
    metric = log_loss(y_test, class_pred)

    print 'The log loss is: ', metric
    if metric < best_metric:
        best_metric = metric
        best_params = params
    print 'The best metric is:', best_metric, 'for the params:', best_params

# The best metric is: 0.788417796191 for the params:  {'num_class': 38, 'silent': 1, 'eval_metric': 'mlogloss', 'nthread': 4, 'objective': 'multi:softprob', 'eta': 0.1, 'num_round': 100, 'max_depth': 7, 'chi2_lim': 250}
# The best metric is:  0.761735967063 for the params:  {'num_class': 38, 'silent': 1, 'eval_metric': 'mlogloss', 'nthread': 4, 'objective': 'multi:softprob', 'eta': 0.1, 'num_round': 200, 'max_depth': 7, 'chi2_lim': 0}
# train-mlogloss:0.339249

# {'num_class': 38, 'silent': 1, 'eval_metric': 'mlogloss', 'nthread': 4, 'objective': 'multi:softprob', 'eta': 0.1, 'num_round': 300, 'max_depth': 6, 'chi2_lim': 0}
# 1381  chi2 columns
# start CV
# [0]	train-mlogloss:2.778163	test-mlogloss:2.804167
# [1]	train-mlogloss:2.459369	test-mlogloss:2.498193
# [2]	train-mlogloss:2.241110	test-mlogloss:2.289825
# [3]	train-mlogloss:2.071988	test-mlogloss:2.128342
# [4]	train-mlogloss:1.933007	test-mlogloss:1.996711
# [5]	train-mlogloss:1.816016	test-mlogloss:1.886429
# [6]	train-mlogloss:1.716173	test-mlogloss:1.792143
# [7]	train-mlogloss:1.628776	test-mlogloss:1.710052
# [8]	train-mlogloss:1.551727	test-mlogloss:1.638188
# [9]	train-mlogloss:1.483332	test-mlogloss:1.574346
# [10]	train-mlogloss:1.421724	test-mlogloss:1.517516
# [11]	train-mlogloss:1.366590	test-mlogloss:1.466812
# [12]	train-mlogloss:1.315932	test-mlogloss:1.420337
# [13]	train-mlogloss:1.270089	test-mlogloss:1.378664
# [14]	train-mlogloss:1.228127	test-mlogloss:1.340715
# [15]	train-mlogloss:1.189455	test-mlogloss:1.305895