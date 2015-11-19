from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.feature_selection import chi2
from sklearn.linear_model import SGDClassifier

__author__ = 'YBeer'

train_result = pd.DataFrame.from_csv("train_result.csv")
col = list(train_result.columns.values)
result_ind = list(train_result[col[0]].value_counts().index)
# train_result = pd.get_dummies(train_result)
train_result = np.array(train_result).ravel()

train = pd.DataFrame.from_csv("train_dummied_150_sep_dep_fln_b_r_v5.csv").astype('float')
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
# param_grid = {'loss': ['log', 'modified_huber'], 'alpha': [0.1], 'n_iter': [200], 'chi2_lim': [1000],
#               'penalty': ['elasticnet'], 'l1_ratio': [0.15], 'n_jobs': [3]}
param_grid = {'loss': ['modified_huber'], 'alpha': [0.003, 0.01, 0.03], 'n_iter': [200], 'chi2_lim': [1000],
              'penalty': ['elasticnet'], 'l1_ratio': [0.15], 'n_jobs': [3]}

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
    cv_n = 4
    kf = StratifiedKFold(train_result, n_folds=cv_n, shuffle=True)
    metric = []
    meta_estimator = np.zeros((train_arr.shape[0], 38))
    for train_index, test_index in kf:
        X_train, X_test = train_arr[train_index, :], train_arr[test_index, :]
        y_train, y_test = train_result[train_index].ravel(), train_result[test_index].ravel()

        classifier = SGDClassifier(loss=params['loss'], alpha=params['alpha'], n_iter=params['n_iter'],
                                   penalty=params['penalty'], l1_ratio=params['l1_ratio'], n_jobs=params['n_jobs'])
        classifier.fit(X_train, y_train)

        # predict
        class_pred = classifier.predict_proba(X_test)

        meta_estimator[test_index, :] = class_pred

        # evaluate
        # print log_loss(y_test, class_pred)
        metric = log_loss(y_test, class_pred)

    print 'The log loss is: ', metric
    if metric < best_metric:
        best_metric = metric
        best_params = params
    print 'The best metric is:', best_metric, 'for the params:', best_params

    meta_estimator = pd.DataFrame(meta_estimator)
    meta_estimator.index = train.index
    # meta_estimator.to_csv('meta_SDG_' + params['loss'] + '.csv')

