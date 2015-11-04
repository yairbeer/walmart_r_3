from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
param_grid = {'n_estimators': [100], 'max_features': [.05, .1, 0.2], 'max_depth': [30, 50, 70],
              'min_samples_split': [1], 'min_samples_leaf': [1]}

for params in ParameterGrid(param_grid):
    print params
    classifier = RandomForestClassifier(n_estimators=params['n_estimators'], max_features=params['max_features'],
                                        max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],
                                        min_samples_leaf=params['min_samples_leaf'])

    # CV
    cv_n = 4
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

# {'max_features': 0.02, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  1.17390386121
# The best metric is:  1.17390386121 for the params:  {'max_features': 0.02, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.02, 'min_samples_split': 15, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  1.1897201987
# The best metric is:  1.17390386121 for the params:  {'max_features': 0.02, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.02, 'min_samples_split': 25, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  1.19948399053
# The best metric is:  1.17390386121 for the params:  {'max_features': 0.02, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.04, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  1.01057948405
# The best metric is:  1.01057948405 for the params:  {'max_features': 0.04, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.04, 'min_samples_split': 15, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  1.02737573519
# The best metric is:  1.01057948405 for the params:  {'max_features': 0.04, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.04, 'min_samples_split': 25, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  1.03890821182
# The best metric is:  1.01057948405 for the params:  {'max_features': 0.04, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.06, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  0.959270520459
# The best metric is:  0.959270520459 for the params:  {'max_features': 0.06, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.06, 'min_samples_split': 15, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  0.972323060292
# The best metric is:  0.959270520459 for the params:  {'max_features': 0.06, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.06, 'min_samples_split': 25, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  0.985040946822
# The best metric is:  0.959270520459 for the params:  {'max_features': 0.06, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.08, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  0.940834661624
# The best metric is:  0.940834661624 for the params:  {'max_features': 0.08, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.08, 'min_samples_split': 15, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  0.94969992212
# The best metric is:  0.940834661624 for the params:  {'max_features': 0.08, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.08, 'min_samples_split': 25, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  0.960915030293
# The best metric is:  0.940834661624 for the params:  {'max_features': 0.08, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.02, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 60, 'min_samples_leaf': 1}
# {'max_features': 0.04, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  1.01148150314
# The best metric is:  1.01148150314 for the params:  {'max_features': 0.04, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.04, 'min_samples_split': 15, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  1.02605725969
# The best metric is:  1.01148150314 for the params:  {'max_features': 0.04, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.06, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  0.958893388618
# The best metric is:  0.958893388618 for the params:  {'max_features': 0.06, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.06, 'min_samples_split': 15, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  0.973724919189
# The best metric is:  0.958893388618 for the params:  {'max_features': 0.06, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.08, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  0.940082522507
# The best metric is:  0.940082522507 for the params:  {'max_features': 0.08, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.08, 'min_samples_split': 15, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# The log loss is:  0.948637860883
# The best metric is:  0.940082522507 for the params:  {'max_features': 0.08, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.04, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 60, 'min_samples_leaf': 1}
# The log loss is:  0.971201777159
# The best metric is:  0.940082522507 for the params:  {'max_features': 0.08, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.04, 'min_samples_split': 15, 'n_estimators': 500, 'max_depth': 60, 'min_samples_leaf': 1}
# The log loss is:  0.987032192516
# The best metric is:  0.940082522507 for the params:  {'max_features': 0.08, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 1}
# {'max_features': 0.06, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 60, 'min_samples_leaf': 1}
# The log loss is:  0.937271324107
# The best metric is:  0.937271324107 for the params:  {'max_features': 0.06, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 60, 'min_samples_leaf': 1}
# {'max_features': 0.06, 'min_samples_split': 15, 'n_estimators': 500, 'max_depth': 60, 'min_samples_leaf': 1}
# The log loss is:  0.945335706754
# The best metric is:  0.937271324107 for the params:  {'max_features': 0.06, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 60, 'min_samples_leaf': 1}
# {'max_features': 0.08, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 60, 'min_samples_leaf': 1}
# The log loss is:  0.921276271806
# The best metric is:  0.921276271806 for the params:  {'max_features': 0.08, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 60, 'min_samples_leaf': 1}
# {'max_features': 0.08, 'min_samples_split': 15, 'n_estimators': 500, 'max_depth': 60, 'min_samples_leaf': 1}
# The log loss is:  0.930886485009
# The best metric is:  0.921276271806 for the params:  {'max_features': 0.08, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 60, 'min_samples_leaf': 1}
# {'max_features': 0.04, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 70, 'min_samples_leaf': 1}
# The log loss is:  0.952010767264
# The best metric is:  0.921276271806 for the params:  {'max_features': 0.08, 'min_samples_split': 5, 'n_estimators': 500, 'max_depth': 60, 'min_samples_leaf': 1}
# {'max_features': 0.04, 'min_samples_split': 15, 'n_estimators': 500, 'max_depth': 70, 'min_samples_leaf': 1}