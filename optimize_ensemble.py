from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
__author__ = 'WiBeer'

"""
data ML
"""
train = pd.DataFrame.from_csv("train_dummied_200_sep_dep_fln_b_r.csv")
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

classifier = RandomForestClassifier(n_estimators=500, max_features=0.06, max_depth=60, min_samples_split=25,
                                    min_samples_leaf=1)
# PCA
pcaing = PCA(n_components=200)
train_pca = pcaing.fit_transform(train)

classifier_pca = RandomForestClassifier(n_estimators=500, max_features=0.4, max_depth=20, min_samples_split=15,
                                        min_samples_leaf=1)

# CV
cv_n = 4
kf = StratifiedKFold(train_result, n_folds=cv_n, shuffle=True)

# prepare dataset for ensemble
metric = []
ensemble_pred = []
ensemble_pred_results = []
for train_index, test_index in kf:
    X_train, X_test = train[train_index, :], train[test_index, :]
    X_train_pca, X_test_pca = train_pca[train_index, :], train_pca[test_index, :]
    y_train, y_test = train_result[train_index].ravel(), train_result[test_index].ravel()
    # train machine learning
    classifier.fit(X_train, y_train)
    classifier_pca.fit(X_train_pca, y_train)

    # predict
    class_pred = classifier.predict_proba(X_test)
    class_pred_pca = classifier_pca.predict_proba(X_test_pca)

    cur_ensemble_pred = np.hstack((class_pred, class_pred_pca))
    ensemble_pred.append(cur_ensemble_pred)
    ensemble_pred_results.append(y_test.reshape((y_test.shape[0], 1)))

ensemble_pred = tuple(ensemble_pred)
ensemble_pred = pd.DataFrame(np.vstack(ensemble_pred))
ensemble_pred.to_csv('ensemble_train_opt.csv')

print type(ensemble_pred_results[0])
ensemble_pred_results = tuple(ensemble_pred_results)
ensemble_pred_results = pd.DataFrame(np.vstack(ensemble_pred_results))
ensemble_pred_results.to_csv('ensemble_train_opt_results.csv')

"""
Ensemble analysis
"""
ensemble_x = pd.DataFrame.from_csv('ensemble_train_opt.csv')
ensemble_x = np.array(ensemble_x)
ensemble_y = pd.DataFrame.from_csv('ensemble_train_opt_results.csv')
ensemble_y = np.array(ensemble_y).ravel()

param_grid = {'penalty': ['l1', 'l2'], 'solver': ['newton-cg', 'lbfgs'],
              'multi_class': ['multinomial', 'ovr'],
              'max_iter': [400]}

# CV
cv_n = 4
kf = StratifiedKFold(ensemble_y, n_folds=cv_n, shuffle=True)

best_metric = 10
best_params = []
for params in ParameterGrid(param_grid):
    print params
    ensemble_log = LogisticRegression(penalty=params['penalty'], solver=params['solver'],
                                      multi_class=params['multi_class'], max_iter=params['max_iter'])

    metric = []
    for train_index, test_index in kf:
        X_train, X_test = ensemble_x[train_index, :], ensemble_x[test_index, :]
        y_train, y_test = ensemble_y[train_index].ravel(), ensemble_y[test_index].ravel()

        # train machine learning
        ensemble_log.fit(X_train, y_train)

        # predict
        class_pred = ensemble_log.predict_proba(X_test)
        metric.append(log_loss(y_test, class_pred))

    print 'The log loss is: ', np.mean(metric)
    if np.mean(metric) < best_metric:
        best_metric = np.mean(metric)
        best_params = params

print 'The best metric is: ', best_metric, 'for the params: ', best_params
