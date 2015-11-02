import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
__author__ = 'WiBeer'

"""
data ML
"""
print 'read training data'
train = pd.DataFrame.from_csv("train_dummied_200_sep_dep_fln_b_r.csv")
train_result = np.array(pd.DataFrame.from_csv("train_result.csv")).ravel()
train = np.array(train)

# Common preprocessing
# Standardizing
stding = StandardScaler()
train = stding.fit_transform(train)

classifier = RandomForestClassifier(n_estimators=1000, max_features=0.06, max_depth=60, min_samples_split=15,
                                    min_samples_leaf=1)

# predict testset
print 'Fitting'
classifier.fit(train, train_result)

print 'read test data'
test = pd.DataFrame.from_csv("test_dummied_200_sep_dep_fln_b_r.csv")
test = np.array(test)
test = stding.transform(test)

predicted_results = classifier.predict_proba(test)

print 'writing to file'
submission_file = pd.DataFrame.from_csv("sample_submission.csv")
submission_file[list(submission_file.columns.values)] = predicted_results
submission_file.to_csv("dep_fln_upc_200_opt_RF_sep_d_fln.csv")

# knn n_neighbors=400, n_components = 10, 1.7

