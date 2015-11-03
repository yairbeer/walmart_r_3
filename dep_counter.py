import pandas as pd
import numpy as np

__author__ = 'WiBeer'


def add_prefix(dataset, prefix):
    col_names = np.array(dataset.columns.values).astype('str')
    for i in range(col_names.shape[0]):
        col_names[i] = prefix + col_names[i]
    dataset.columns = col_names
    return dataset


def remove_sparse(dataset):
    col_names = np.array(dataset.columns.values).astype('str')
    dead_col = []
    for i in range(col_names.shape[0]):
        if not np.sum(dataset[col_names[i]]):
            dead_col.append(i)
    dataset = dataset.drop(dataset.columns[dead_col], 1)
    return dataset

"""
preprocessing data
"""
# preprocess test data
print 'read train data'
trainset = pd.DataFrame.from_csv('train.csv', index_col=1)
trainset = trainset.fillna('-999')
trainset[['Upc', 'FinelineNumber']] = trainset[['Upc', 'FinelineNumber']].astype(str)
n = trainset.shape[0]

train_result = trainset['TripType']
train_result = train_result.groupby(by=train_result.index, sort=False).mean()

dep_count = train_result
dep_count.columns = ['dep_num']

# indexes = list(trainset.index.values)
# for i in range(len(indexes)):
#     single_vis = trainset.loc[indexes[i]]
#     if type(single_vis['DepartmentDescription']) == 'str':
#         dep_count.loc[indexes[i]] = 1
#     else:
#         print single_vis['DepartmentDescription'].value_counts()
#         print len(list(single_vis['DepartmentDescription'].value_counts()))

n_trips = train_result.shape[0]

train_data_not_count = pd.get_dummies(trainset['Weekday'])
train_data_not_count = train_data_not_count.groupby(by=train_data_not_count.index, sort=False).mean()

# separate between returned and bought goods
train_total_items = np.array(trainset['ScanCount']).reshape((n, 1))
train_bought_items = np.clip(train_total_items, a_min=0, a_max=99999)
train_returned_items = np.clip(train_total_items, a_min=-99999, a_max=0)

print 'dummy train DepartmentDescription'
train_data_count_dep = pd.get_dummies(trainset['DepartmentDescription'])
tmp_index = train_data_count_dep.index
tmp_columns = list(train_data_count_dep.columns.values)

# separate b/r department variables
# bought Department
tmp_table = np.array(train_data_count_dep) * train_bought_items
train_data_count_dep_bought = pd.DataFrame(tmp_table)
train_data_count_dep_bought.columns = tmp_columns
train_data_count_dep_bought.index = tmp_index
train_data_count_dep_bought = train_data_count_dep_bought.groupby(by=train_data_count_dep_bought.index, sort=False).sum()
train_data_count_dep_bought = add_prefix(train_data_count_dep_bought, 'Dep_B_')

train_num_deps_b = (np.array(train_data_count_dep_bought) > 0)
train_num_deps_b = pd.DataFrame(np.sum(train_num_deps_b, axis=1))
train_num_deps_b.columns = ['Deps_num_B']
train_num_deps_b.index = train_data_count_dep_bought.index
print train_num_deps_b

# returned Department
tmp_table = np.array(train_data_count_dep) * train_returned_items
train_data_count_dep_returned = pd.DataFrame(tmp_table)
train_data_count_dep_returned.columns = tmp_columns
train_data_count_dep_returned.index = tmp_index
train_data_count_dep_returned = train_data_count_dep_returned.groupby(by=train_data_count_dep_returned.index, sort=False).sum()
train_data_count_dep_returned = add_prefix(train_data_count_dep_returned, 'Dep_R_')

train_num_deps_r = (np.array(train_data_count_dep_returned) < 0)
train_num_deps_r = pd.DataFrame(np.sum(train_num_deps_r, axis=1))
train_num_deps_r.columns = ['Deps_num_R']
train_num_deps_r.index = train_data_count_dep_bought.index
print train_num_deps_r
