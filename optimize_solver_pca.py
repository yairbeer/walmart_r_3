from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
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
param_grid = {'pca_est': [200], 'n_estimators': [100], 'max_features': [0.1, 0.2, 0.4], 'max_depth': [10, 20, 30],
              'min_samples_split': [1], 'min_samples_leaf': [1]}

for params in ParameterGrid(param_grid):
    print params
    classifier = RandomForestClassifier(n_estimators=params['n_estimators'], max_features=params['max_features'],
                                        max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],
                                        min_samples_leaf=params['min_samples_leaf'])

    # PCA
    pcaing = PCA(n_components=params['pca_est'])
    train_pca = pcaing.fit_transform(train)

    # CV
    cv_n = 4
    kf = StratifiedKFold(train_result, n_folds=cv_n, shuffle=True)

    metric = []
    for train_index, test_index in kf:
        X_train, X_test = train_pca[train_index, :], train_pca[test_index, :]
        y_train, y_test = train_result[train_index].ravel(), train_result[test_index].ravel()
        # train machine learning
        classifier.fit(X_train, y_train)

        # predict
        class_pred = classifier.predict_proba(X_test)

        # evaluate
        metric.append(log_loss(y_test, class_pred))

    print 'The log loss is: ', np.mean(metric)
    if np.mean(metric) < best_metric:
        best_metric = np.mean(metric)
        best_params = params
    print 'The best metric is: ', best_metric, 'for the params: ', best_params
