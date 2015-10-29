import pandas as pd
import numpy as np
import random
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
__author__ = 'WiBeer'

"""
data ML
"""
train = pd.DataFrame.from_csv("train_dummied_001_sep_b_r.csv")
train_result = np.array(pd.DataFrame.from_csv("train_result.csv")).ravel()
test = pd.DataFrame.from_csv("test_dummied_001_sep_b_r.csv")
train = np.array(train)
test = np.array(test)

# Common preprocessing
# Standardizing
stding = StandardScaler()
train = stding.fit_transform(train)
test = stding.transform(test)

# # PCA
# pcaing = PCA(n_components=100)
# train = pcaing.fit_transform(train)
# test = pcaing.transform(test)

classifier = RandomForestClassifier(n_estimators=2000, max_features=0.1, max_depth=32, min_samples_split=7,
                                    min_samples_leaf=1)

# # CV
# cv_n = 4
# kf = StratifiedKFold(train_result, n_folds=cv_n, shuffle=True)
#
# print 'start CV'
# metric = []
# for train_index, test_index in kf:
#     X_train, X_test = train[train_index, :], train[test_index, :]
#     y_train, y_test = train_result[train_index].ravel(), train_result[test_index].ravel()
#     # train machine learning
#     classifier.fit(X_train, y_train)
#
#     # predict
#     class_pred = classifier.predict_proba(X_test)
#
#     # evaluate
#     print log_loss(y_test, class_pred)
#     metric.append(log_loss(y_test, class_pred))
#
# print 'The log loss is: ', np.mean(metric)

# predict testset
classifier.fit(train, train_result)
predicted_results = classifier.predict_proba(test)

submission_file = pd.DataFrame.from_csv("sample_submission.csv")
submission_file[list(submission_file.columns.values)] = predicted_results
submission_file.to_csv("dep_fln_upc_001_coarse_opt_RF.csv")

# knn n_neighbors=400, n_components = 10, 1.7

