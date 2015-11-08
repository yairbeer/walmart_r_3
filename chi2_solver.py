from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.feature_selection import chi2
from sklearn.ensemble import GradientBoostingClassifier


__author__ = 'YBeer'

train = pd.DataFrame.from_csv("train_dummied_200_sep_dep_fln_b_r_v2.csv")
train.fillna(0)
train_arr = np.array(train)
col_list = list(train.columns.values)

train_result = pd.DataFrame.from_csv("train_result.csv")
# train_result = pd.get_dummies(train_result)
train_result = np.array(train_result).ravel()

# print train_result.shape[1], ' categorial'
print train.shape[1], ' columns'

print 'absing'
for i in range(train_arr.shape[0]):
    for j in range(train_arr.shape[1]):
        train_arr[i, j] = np.abs(train_arr[i, j])

chi2_params = chi2(train_arr, train_result)

# for i in range(train_arr.shape[1]):
#     print col_list[i], chi2_params[0][i]

# filtering low chi2 cols
chi2_lim = 1000
chi2_cols = []
for i in range(train.shape[1]):
    if chi2_params[0][i] > chi2_lim:
        chi2_cols.append(col_list[i])

del train_arr

print len(chi2_cols), ' chi2 columns'
train = train[chi2_cols]
train = np.array(train)

# Standardizing
stding = StandardScaler()
train = stding.fit_transform(train)

print 'start fitting'

classifier = GradientBoostingClassifier(n_estimators=200, max_depth=5, max_features=0.6, learning_rate=0.075)

classifier.fit(train, train_result)

test = pd.DataFrame.from_csv("test_dummied_200_sep_dep_fln_b_r_v2.csv")
test.fillna(0)
test = test[chi2_cols]
test = np.array(test)
test = stding.transform(test)

predicted_results = classifier.predict_proba(test)

print 'writing to file'
submission_file = pd.DataFrame.from_csv("sample_submission.csv")
submission_file[list(submission_file.columns.values)] = predicted_results
submission_file.to_csv("chi2_feature_select_2.csv")
# predict
class_pred = classifier.predict_proba(test)

# evaluate
# print log_loss(y_test, class_pred)
print log_loss(train_result, classifier.predict_proba(train))

