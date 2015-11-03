from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from random import seed

__author__ = 'WiBeer'

"""
data ML
"""
train = pd.DataFrame.from_csv("train_dummied_200_sep_dep_b_r.csv")
train_result = np.array(pd.DataFrame.from_csv("train_result.csv")).ravel()
train = np.array(train)

# Common preprocessing
# Standardizing
stding = StandardScaler()
train = stding.fit_transform(train)

# # PCA
# pcaing = PCA(n_components=100)
# train = pcaing.fit_transform(train)
# test = pcaing.transform(test)

print 'start CV'
best_metric = 10
best_params = []
param_grid = {'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'n_estimators': [100], 'max_features': [.06],
              'max_depth': [70], 'min_samples_split': [15], 'min_samples_leaf': [1]}

for params in ParameterGrid(param_grid):
    print params
    classifier = RandomForestClassifier(n_estimators=params['n_estimators'], max_features=params['max_features'],
                                        max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],
                                        min_samples_leaf=params['min_samples_leaf'])

    new_train = train
    seed(1)
    cluster = KMeans(n_clusters=params['n_clusters'])
    cluster.fit(train)
    # print 'The cluster\'s inertia is ', cluster.inertia_
    # for i in range(params['n_clusters']):
    #     tmp_series = train_result.iloc[cluster.labels_ == i]
    #     print 'group', i, tmp_series.shape
    new_col = pd.DataFrame(cluster.labels_).astype('str')
    new_col.columns = ['kmeans_' + str(params['n_clusters'])]
    new_col = np.array(pd.get_dummies(new_col))
    new_train = np.hstack((new_train, new_col))
    print new_train.shape[1], ' columns'


    # CV
    cv_n = 4
    kf = StratifiedKFold(train_result, n_folds=cv_n, shuffle=True)

    metric = []
    for train_index, test_index in kf:
        X_train, X_test = new_train[train_index, :], new_train[test_index, :]
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