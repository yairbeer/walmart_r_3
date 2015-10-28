import pandas as pd
import numpy as np
import random
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold

__author__ = 'WiBeer'

"""
data ML
"""
train = pd.DataFrame.from_csv("train_dummied.csv")
train_result = np.array(pd.DataFrame.from_csv("train_result.csv")).ravel()
test = pd.DataFrame.from_csv("test_dummied.csv")
print list(train.columns.values)
print list(test.columns.values)
train = np.array(train)
test = np.array(test)

# Common preprocessing
# Standardizing
stding = StandardScaler()
train = stding.fit_transform(train)
test = stding.transform(test)

classifier = RandomForestClassifier(n_estimators=300)

# # CV n = 4
# cv_n = 4

# metric = []
# for train_index, test_index in kf:
#     X_train, X_test = train_cv[train_index, :], train_cv[test_index, :]
#     y_train, y_test = train_result_cv[train_index].ravel(), train_result_cv[test_index].ravel()
#     print X_train.shape, X_test.shape
#     print len(y_train), len(y_test )
#     # train machine learning
#     classifier.fit(X_train, y_train)

#     # predict
#     class_pred = classifier.predict_proba(X_test)

#     # evaluate
#     print log_loss(y_test, class_pred)
#     metric.append(log_loss(y_test, class_pred))
# print 'The log loss is: ', np.mean(metric)

# predict testset
classifier.fit(train, train_result)
predicted_results = classifier.predict_proba(test)

predicted_results_self = classifier.predict_proba(train)
print log_loss(train_result, predicted_results_self)

# # PCA
# pcaing = PCA(n_components=2)
# train_pca = pcaing.fit_transform(train)
# test_pca = pcaing.transform(test)

submission_file = pd.DataFrame.from_csv("sample_submission.csv")
submission_file[list(submission_file.columns.values)] = predicted_results
submission_file.to_csv("dep_fln002_rf.csv")
