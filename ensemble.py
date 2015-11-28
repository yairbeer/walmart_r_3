from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
train = glob.glob('meta_train*')
print train
for i in range(len(train)):
    train[i] = pd.DataFrame.from_csv(train[i])
train = pd.concat(train, axis=1)

test = glob.glob('meta_test*')
print test
for i in range(len(test)):
    test[i] = pd.DataFrame.from_csv(test[i])
test = pd.concat(test, axis=1)

# print train_result.shape[1], ' categorial'
print train.shape[1], ' columns'

best_metric = 10
best_params = []
param_grid = {'num_class': 38, 'silent': 1, 'eval_metric': 'mlogloss', 'subsample': 0.5, 'nthread': 3,
              'objective': 'multi:softprob', 'eta': 0.03, 'num_round': 400, 'max_depth': 5, 'chi2_lim': 0}


for params in ParameterGrid(param_grid):
    print params

    # Standardizing
    stding = StandardScaler()
    train = stding.fit_transform(train)
    test = stding.transform(test)

    # train machine learning
    xg_train = xgboost.DMatrix(train, label=train_result_xgb)
    xg_test = xgboost.DMatrix(test)

    watchlist = [(xg_train, 'train')]

    num_round = params['num_round']
    xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);

    # predict
    predicted_results = xgclassifier.predict(xg_test)
    predicted_results = predicted_results.reshape(test.shape[0], 38)

    print 'writing to file'
    submission_file = pd.DataFrame.from_csv("sample_submission.csv")
    submission_cols = list(submission_file.columns.values)
    submission_vals = map(lambda x: int(x.split("_")[1]), submission_cols)

    submission_table = np.zeros(predicted_results.shape)
    for i in range(predicted_results.shape[1]):
        for j in range(predicted_results.shape[1]):
            if submission_vals[i] == result_ind[j]:
                print 'adding triptype ', submission_vals[i]
                submission_table[:, i] = predicted_results[:, j]

    submission_file[list(submission_file.columns.values)] = submission_table
    submission_file.to_csv("ensemble_xgboost_opt.csv")
