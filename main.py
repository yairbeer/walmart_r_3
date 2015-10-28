import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold

__author__ = 'WiBeer'

# preprocess test data
trainset = pd.DataFrame.from_csv('train.csv', index_col=1)
trainset = trainset.fillna('-999')
trainset[['Upc', 'FinelineNumber']] = trainset[['Upc', 'FinelineNumber']].astype(str)
n = trainset.shape[0]

train_result = trainset['TripType']
train_result = train_result.groupby(by=train_result.index, sort=False).mean()
n_trips = train_result.shape[0]

train_data_not_count = pd.get_dummies(trainset['Weekday'])
train_data_not_count = train_data_not_count.groupby(by=train_data_not_count.index, sort=False).mean()

train_data_count_dep = pd.get_dummies(trainset['DepartmentDescription'])
tmp_index = train_data_count_dep.index
tmp_columns = list(train_data_count_dep.columns.values)
tmp_table = np.array(train_data_count_dep) * np.array(trainset['ScanCount']).reshape((n, 1))
train_data_count_dep = pd.DataFrame(tmp_table)
train_data_count_dep.columns = tmp_columns
train_data_count_dep.index = tmp_index
train_data_count_dep = train_data_count_dep.groupby(by=train_data_count_dep.index, sort=False).sum()

# need to separate between returned and bought goods

train_data_tot_items = trainset['ScanCount'].groupby(by=trainset.index, sort=False).sum()

# find most bought FinelineNumber
fineline_density = trainset['FinelineNumber'].value_counts()
sparsity = n_trips * 0.02
n_features = np.sum(fineline_density > sparsity)
# print n_features
fineline_density = fineline_density.iloc[:n_features]
fineline_cols = list(fineline_density.index)
print fineline_cols

# remove sparse FinelineNumber products
tmp_series = np.zeros((trainset.shape[0], 1))
for i in range(trainset.shape[0]):
    flnumber = trainset.iloc[i]['FinelineNumber']
    if flnumber in fineline_cols:
        tmp_series[i] = flnumber
trainset['FinelineNumber'] = tmp_series
print trainset['FinelineNumber'].value_counts()

# dummy fln
train_data_count_fln = pd.get_dummies(trainset['FinelineNumber'])
tmp_index = train_data_count_fln.index
tmp_columns = list(train_data_count_fln.columns.values)
tmp_table = np.array(train_data_count_fln) * np.array(trainset['ScanCount']).reshape((n, 1))
train_data_count_fln = pd.DataFrame(tmp_table)
train_data_count_fln.columns = tmp_columns
train_data_count_fln.index = tmp_index
train_data_count_fln = train_data_count_fln.groupby(by=train_data_count_fln.index, sort=False).sum()

train = pd.concat([train_data_not_count, train_data_count_dep, train_data_count_fln, train_data_tot_items], axis=1)

# preprocess test data
testset = pd.DataFrame.from_csv('test.csv', index_col=0)
test_data_not_count = pd.get_dummies(testset['Weekday'])
test_data_not_count = test_data_not_count.groupby(by=test_data_not_count.index, sort=False).mean()
n_test = testset.shape[0]

test_data_count_dep = pd.get_dummies(testset['DepartmentDescription'])
tmp_index = test_data_count_dep.index
tmp_columns = list(test_data_count_dep.columns.values)
tmp_table = np.array(test_data_count_dep) * np.array(testset['ScanCount']).reshape((n_test, 1))
test_data_count_dep = pd.DataFrame(tmp_table)
test_data_count_dep.columns = tmp_columns
test_data_count_dep.index = tmp_index
test_data_count_dep = test_data_count_dep.groupby(by=test_data_count_dep.index, sort=False).sum()

test_data_tot_items = testset['ScanCount'].groupby(by=testset.index, sort=False).sum()

# find most bought FinelineNumber
fineline_density = testset['FinelineNumber'].value_counts()
sparsity = n_trips * 0.02
n_features = np.sum(fineline_density > sparsity)
# print n_features
fineline_density = fineline_density.iloc[:n_features]
fineline_cols = list(fineline_density.index)
print fineline_cols

# remove sparse FinelineNumber products
tmp_series = np.zeros((testset.shape[0], 1))
for i in range(testset.shape[0]):
    flnumber = testset.iloc[i]['FinelineNumber']
    if flnumber in fineline_cols:
        tmp_series[i] = flnumber
testset['FinelineNumber'] = tmp_series
print testset['FinelineNumber'].value_counts()

# dummy fln
test_data_count_fln = pd.get_dummies(testset['FinelineNumber'])
tmp_index = test_data_count_fln.index
tmp_columns = list(test_data_count_fln.columns.values)
tmp_table = np.array(test_data_count_fln) * np.array(testset['ScanCount']).reshape((n, 1))
test_data_count_fln = pd.DataFrame(tmp_table)
test_data_count_fln.columns = tmp_columns
test_data_count_fln.index = tmp_index
test_data_count_fln = test_data_count_fln.groupby(by=test_data_count_fln.index, sort=False).sum()

test = pd.concat([test_data_not_count, test_data_count_dep, test_data_count_fln, test_data_tot_items], axis=1)

# Find common coloumns
col_train = list(train.columns.values)
col_test = list(test.columns.values)
col_common = []
# add only common columns for train and test
for col in col_train:
    if col in col_test:
        col_common.append(col)
train = train[col_common]
test = test[col_common]

# Common preprocessing
# Standardizing
stding = StandardScaler()
train = stding.fit_transform(train)
test = stding.transform(test)

print col_common

classifier = RandomForestClassifier(n_estimators=200)

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
log_loss(train_result, predicted_results_self)

# # PCA
# pcaing = PCA(n_components=2)
# train_pca = pcaing.fit_transform(train)
# test_pca = pcaing.transform(test)

submission_file = pd.DataFrame.from_csv("sample_submission.csv")
submission_file[list(submission_file.columns.values)] = predicted_results
submission_file.to_csv("trysub_rf.csv")
