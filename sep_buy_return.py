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
print train_result.value_counts()
# train_result.to_csv("train_result.csv")

train_dep_count_b = train_result.copy(deep=True)
train_dep_count_r = train_result.copy(deep=True)

train_fln_count_b = train_result.copy(deep=True)
train_fln_count_r = train_result.copy(deep=True)

train_upc_count_b = train_result.copy(deep=True)
train_upc_count_r = train_result.copy(deep=True)

train_dep_count_b.columns = ['dep_num_B']
train_dep_count_r.columns = ['dep_num_R']

train_fln_count_b.columns = ['fln_num_B']
train_fln_count_r.columns = ['fln_num_R']

train_upc_count_b.columns = ['upc_num_B']
train_upc_count_r.columns = ['upc_num_R']
indexes = list(train_result.index.values)

print 'Department counter'
for i in range(len(indexes)):
    single_vis = trainset.loc[indexes[i]]
    bought = np.array(single_vis['ScanCount'] > 0)
    returned = np.array(single_vis['ScanCount'] < 0)
    single_vis_bought = single_vis.iloc[bought]
    single_vis_returned = single_vis.iloc[returned]

    if single_vis_bought.shape[0] == 0:
        train_dep_count_b.loc[indexes[i]] = 0
    else:
        if len(single_vis_bought.shape) == 1:
            train_dep_count_b.loc[indexes[i]] = 1
        else:
            train_dep_count_b.loc[indexes[i]] = len(list(single_vis_bought['DepartmentDescription'].value_counts()))

    if single_vis_returned.shape[0] == 0:
        train_dep_count_r.loc[indexes[i]] = 0
    else:
        if len(single_vis_returned.shape) == 1:
            train_dep_count_r.loc[indexes[i]] = 1
        else:
            train_dep_count_r.loc[indexes[i]] = len(list(single_vis_returned['DepartmentDescription'].value_counts()))

print 'fln counter'
for i in range(len(indexes)):
    single_vis = trainset.loc[indexes[i]]
    bought = np.array(single_vis['ScanCount'] > 0)
    returned = np.array(single_vis['ScanCount'] < 0)
    single_vis_bought = single_vis.iloc[bought]
    single_vis_returned = single_vis.iloc[returned]

    if single_vis_bought.shape[0] == 0:
        train_fln_count_b.loc[indexes[i]] = 0
    else:
        if len(single_vis_bought.shape) == 1:
            train_fln_count_b.loc[indexes[i]] = 1
        else:
            train_fln_count_b.loc[indexes[i]] = len(list(single_vis_bought['FinelineNumber'].value_counts()))

    if single_vis_returned.shape[0] == 0:
        train_fln_count_r.loc[indexes[i]] = 0
    else:
        if len(single_vis_returned.shape) == 1:
            train_fln_count_r.loc[indexes[i]] = 1
        else:
            train_fln_count_r.loc[indexes[i]] = len(list(single_vis_returned['FinelineNumber'].value_counts()))

print 'upc counter'
for i in range(len(indexes)):
    single_vis = trainset.loc[indexes[i]]
    bought = np.array(single_vis['ScanCount'] > 0)
    returned = np.array(single_vis['ScanCount'] < 0)
    single_vis_bought = single_vis.iloc[bought]
    single_vis_returned = single_vis.iloc[returned]
    
    if single_vis_bought.shape[0] == 0:
        train_upc_count_b.loc[indexes[i]] = 0
    else:
        if len(single_vis_bought.shape) == 1:
            train_upc_count_b.loc[indexes[i]] = 1
        else:
            train_upc_count_b.loc[indexes[i]] = len(list(single_vis_bought['Upc'].value_counts()))
    
    if single_vis_returned.shape[0] == 0:
        train_upc_count_r.loc[indexes[i]] = 0
    else:
        if len(single_vis_returned.shape) == 1:
            train_upc_count_r.loc[indexes[i]] = 1
        else:
            train_upc_count_r.loc[indexes[i]] = len(list(single_vis_returned['Upc'].value_counts()))

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
train_data_count_dep_bought = \
    train_data_count_dep_bought.groupby(by=train_data_count_dep_bought.index, sort=False).sum()
train_data_count_dep_bought = add_prefix(train_data_count_dep_bought, 'Dep_B_')

# returned Department
tmp_table = np.array(train_data_count_dep) * train_returned_items
train_data_count_dep_returned = pd.DataFrame(tmp_table)
train_data_count_dep_returned.columns = tmp_columns
train_data_count_dep_returned.index = tmp_index
train_data_count_dep_returned = \
    train_data_count_dep_returned.groupby(by=train_data_count_dep_returned.index, sort=False).sum()
train_data_count_dep_returned = add_prefix(train_data_count_dep_returned, 'Dep_R_')

train_data_tot_items = trainset['ScanCount'].groupby(by=trainset.index, sort=False).sum()
train_bought_items = pd.DataFrame(train_bought_items.ravel())
train_bought_items.index = trainset.index
train_bought_items.columns = ['Bought']
train_data_bought_items = train_bought_items.groupby(by=trainset.index, sort=False).sum()
train_returned_items = pd.DataFrame(train_returned_items.ravel())
train_returned_items.index = trainset.index
train_returned_items.columns = ['Returned']
train_data_returned_items = train_returned_items.groupby(by=trainset.index, sort=False).sum()

sparsity = 800

# find most bought FinelineNumber
print 'remove sparse train FinelineNumber'
fineline_density = trainset['FinelineNumber'].value_counts()
n_features = np.sum(fineline_density > sparsity)
# print n_features
fineline_density = fineline_density.iloc[:n_features]
fineline_cols = list(fineline_density.index)

# remove sparse FinelineNumber products
tmp_series = np.zeros((trainset.shape[0], 1))
for i in range(trainset.shape[0]):
    flnumber = trainset.iloc[i]['FinelineNumber']
    if flnumber in fineline_cols:
        tmp_series[i] = flnumber
trainset['FinelineNumber'] = tmp_series
print trainset['FinelineNumber'].value_counts()

# dummy fln
print 'dummy train FinelineNumber'

# bought
train_data_count_fln = pd.get_dummies(trainset['FinelineNumber'])
tmp_index = train_data_count_fln.index
tmp_columns = list(train_data_count_fln.columns.values)
tmp_table = np.array(train_data_count_fln) * np.array(train_bought_items)
train_data_count_fln_bought = pd.DataFrame(tmp_table)
train_data_count_fln_bought.columns = tmp_columns
train_data_count_fln_bought.index = tmp_index
train_data_count_fln_bought = train_data_count_fln_bought.groupby(by=train_data_count_fln_bought.index, sort=False).sum()
train_data_count_fln_bought = add_prefix(train_data_count_fln_bought, 'FLN_B_')

# returned
train_data_count_fln = pd.get_dummies(trainset['FinelineNumber'])
tmp_index = train_data_count_fln.index
tmp_columns = list(train_data_count_fln.columns.values)
tmp_table = np.array(train_data_count_fln) * np.array(train_returned_items)
train_data_count_fln_returned = pd.DataFrame(tmp_table)
train_data_count_fln_returned.columns = tmp_columns
train_data_count_fln_returned.index = tmp_index
train_data_count_fln_returned = train_data_count_fln_returned.groupby(by=train_data_count_fln_returned.index, sort=False).sum()
train_data_count_fln_returned = add_prefix(train_data_count_fln_returned, 'FLN_R_')

# find most bought Upc
print 'remove sparse train Upc'
upc_density = trainset['Upc'].value_counts()
n_features = np.sum(upc_density > sparsity)
# print n_features
upc_density = upc_density.iloc[:n_features]
upc_density = list(upc_density.index)

# remove sparse Upc
tmp_series = np.zeros((trainset.shape[0], 1))
for i in range(trainset.shape[0]):
    upc_number = trainset.iloc[i]['Upc']
    if upc_number in upc_density:
        tmp_series[i] = upc_number
trainset['Upc'] = tmp_series
print trainset['Upc'].value_counts()

# dummy Upc
print 'dummy train Upc'
train_data_count_upc = pd.get_dummies(trainset['Upc'])
tmp_index = train_data_count_upc.index
tmp_columns = list(train_data_count_upc.columns.values)
tmp_table = np.array(train_data_count_upc) * train_total_items
train_data_count_upc = pd.DataFrame(tmp_table)
train_data_count_upc.columns = tmp_columns
train_data_count_upc.index = tmp_index
train_data_count_upc = train_data_count_upc.groupby(by=train_data_count_upc.index, sort=False).sum()
train_data_count_upc = add_prefix(train_data_count_upc, 'Upc_')

train = pd.concat([train_data_not_count, train_data_count_dep_bought, train_data_count_dep_returned,
                   train_data_count_fln_bought, train_data_count_fln_returned,
                   train_data_count_upc, train_data_bought_items, train_data_returned_items,
                   train_dep_count_b, train_dep_count_r, train_fln_count_b, train_fln_count_r,
                   train_upc_count_b, train_upc_count_r, train_data_bought_items, train_data_returned_items], axis=1)
# train = remove_sparse(train)

# preprocess test data
print 'read test data'
testset = pd.DataFrame.from_csv('test.csv', index_col=0)
testset = testset.fillna('-999')

test_data_not_count = pd.get_dummies(testset['Weekday'])
test_data_not_count = test_data_not_count.groupby(by=test_data_not_count.index, sort=False).mean()

n_test = testset.shape[0]
n_trips_test = test_data_not_count.shape[0]

test_total_items = np.array(testset['ScanCount']).reshape((n_test, 1))
test_bought_items = np.clip(test_total_items, a_min=0, a_max=99999)
test_returned_items = np.clip(test_total_items, a_min=-99999, a_max=0)

test_data_count_dep = pd.get_dummies(testset['DepartmentDescription'])
tmp_index = test_data_count_dep.index
tmp_columns = list(test_data_count_dep.columns.values)

# bought department
tmp_table = np.array(test_data_count_dep) * test_bought_items
test_data_count_dep_bought = pd.DataFrame(tmp_table)
test_data_count_dep_bought.columns = tmp_columns
test_data_count_dep_bought.index = tmp_index
test_data_count_dep_bought = test_data_count_dep_bought.groupby(by=test_data_count_dep_bought.index, sort=False).sum()
test_data_count_dep_bought = add_prefix(test_data_count_dep_bought, 'Dep_B_')

# returned department
tmp_table = np.array(test_data_count_dep) * test_returned_items
test_data_count_dep_returned = pd.DataFrame(tmp_table)
test_data_count_dep_returned.columns = tmp_columns
test_data_count_dep_returned.index = tmp_index
test_data_count_dep_returned = test_data_count_dep_returned.groupby(by=test_data_count_dep_returned.index, sort=False).sum()
test_data_count_dep_returned = add_prefix(test_data_count_dep_returned, 'Dep_R_')

test_data_tot_items = testset['ScanCount'].groupby(by=testset.index, sort=False).sum()
test_bought_items = pd.DataFrame(test_bought_items.ravel())
test_bought_items.index = testset.index
test_bought_items.columns = ['Bought']
test_data_bought_items = test_bought_items.groupby(by=testset.index, sort=False).sum()
test_returned_items = pd.DataFrame(test_returned_items.ravel())
test_returned_items.index = testset.index
test_returned_items.columns = ['Returned']
test_data_returned_items = test_returned_items.groupby(by=testset.index, sort=False).sum()

# find most bought FinelineNumber
fineline_density = testset['FinelineNumber'].value_counts()
n_features = np.sum(fineline_density > sparsity)
# print n_features
fineline_density = fineline_density.iloc[:n_features]
fineline_cols = list(fineline_density.index)

# remove sparse FinelineNumber products
tmp_series = np.zeros((testset.shape[0], 1))
for i in range(testset.shape[0]):
    flnumber = testset.iloc[i]['FinelineNumber']
    if flnumber in fineline_cols:
        tmp_series[i] = flnumber
testset['FinelineNumber'] = tmp_series
print testset['FinelineNumber'].value_counts()

# dummy fln
# bought
test_data_count_fln = pd.get_dummies(testset['FinelineNumber'])
tmp_index = test_data_count_fln.index
tmp_columns = list(test_data_count_fln.columns.values)
tmp_table = np.array(test_data_count_fln) * np.array(test_bought_items)
test_data_count_fln_bought = pd.DataFrame(tmp_table)
test_data_count_fln_bought.columns = tmp_columns
test_data_count_fln_bought.index = tmp_index
test_data_count_fln_bought = test_data_count_fln_bought.groupby(by=test_data_count_fln_bought.index, sort=False).sum()
test_data_count_fln_bought = add_prefix(test_data_count_fln_bought, 'FLN_B_')

# returned
test_data_count_fln = pd.get_dummies(testset['FinelineNumber'])
tmp_index = test_data_count_fln.index
tmp_columns = list(test_data_count_fln.columns.values)
tmp_table = np.array(test_data_count_fln) * np.array(test_total_items)
test_data_count_fln_returned = pd.DataFrame(tmp_table)
test_data_count_fln_returned.columns = tmp_columns
test_data_count_fln_returned.index = tmp_index
test_data_count_fln_returned = test_data_count_fln_returned.groupby(by=test_data_count_fln_returned.index, sort=False).sum()
test_data_count_fln_returned = add_prefix(test_data_count_fln_returned, 'FLN_R_')

# find most bought Upc
print 'remove sparse test Upc'
upc_density = testset['Upc'].value_counts()
# print n_features
n_features = np.sum(upc_density > sparsity)
upc_density = upc_density.iloc[:n_features]
upc_density = list(upc_density.index)

# remove sparse Upc
tmp_series = np.zeros((testset.shape[0], 1))
for i in range(testset.shape[0]):
    upc_number = testset.iloc[i]['Upc']
    if upc_number in upc_density:
        tmp_series[i] = upc_number
testset['Upc'] = tmp_series
print testset['Upc'].value_counts()

# dummy Upc
print 'dummy test Upc'
test_data_count_upc = pd.get_dummies(testset['Upc'])
tmp_index = test_data_count_upc.index
tmp_columns = list(test_data_count_upc.columns.values)
tmp_table = np.array(test_data_count_upc) * test_total_items
test_data_count_upc = pd.DataFrame(tmp_table)
test_data_count_upc.columns = tmp_columns
test_data_count_upc.index = tmp_index
test_data_count_upc = test_data_count_upc.groupby(by=test_data_count_upc.index, sort=False).sum()
test_data_count_upc = add_prefix(test_data_count_upc, 'Upc_')

test_dep_count_b = np.ones((test_data_count_fln_bought.shape[0], 1))
test_dep_count_b = pd.DataFrame(test_dep_count_b)
test_dep_count_b.index = test_data_count_upc.index
test_dep_count_r = np.ones((test_data_count_fln_bought.shape[0], 1))
test_dep_count_r = pd.DataFrame(test_dep_count_r)
test_dep_count_r.index = test_data_count_upc.index

test_fln_count_b = np.ones((test_data_count_fln_bought.shape[0], 1))
test_fln_count_b = pd.DataFrame(test_fln_count_b)
test_fln_count_b.index = test_data_count_upc.index
test_fln_count_r = np.ones((test_data_count_fln_bought.shape[0], 1))
test_fln_count_r = pd.DataFrame(test_fln_count_r)
test_fln_count_r.index = test_data_count_upc.index

test_upc_count_b = np.ones((test_data_count_fln_bought.shape[0], 1))
test_upc_count_b = pd.DataFrame(test_upc_count_b)
test_upc_count_b.index = test_data_count_upc.index
test_upc_count_r = np.ones((test_data_count_fln_bought.shape[0], 1))
test_upc_count_r = pd.DataFrame(test_upc_count_r)
test_upc_count_r.index = test_data_count_upc.index

test_dep_count_b.columns = ['dep_num_B']
test_dep_count_r.columns = ['dep_num_R']

test_fln_count_b.columns = ['fln_num_B']
test_fln_count_r.columns = ['fln_num_R']

test_upc_count_b.columns = ['upc_num_B']
test_upc_count_r.columns = ['upc_num_R']
indexes = list(test_data_count_upc.index.values)

print 'Department counter'
for i in range(len(indexes)):
    single_vis = testset.loc[indexes[i]]
    bought = np.array(single_vis['ScanCount'] > 0)
    returned = np.array(single_vis['ScanCount'] < 0)
    single_vis_bought = single_vis.iloc[bought]
    single_vis_returned = single_vis.iloc[returned]

    if single_vis_bought.shape[0] == 0:
        test_dep_count_b.loc[indexes[i]] = 0
    else:
        if len(single_vis_bought.shape) == 1:
            test_dep_count_b.loc[indexes[i]] = 1
        else:
            test_dep_count_b.loc[indexes[i]] = len(list(single_vis_bought['DepartmentDescription'].value_counts()))

    if single_vis_returned.shape[0] == 0:
        test_dep_count_r.loc[indexes[i]] = 0
    else:
        if len(single_vis_returned.shape) == 1:
            test_dep_count_r.loc[indexes[i]] = 1
        else:
            test_dep_count_r.loc[indexes[i]] = len(list(single_vis_returned['DepartmentDescription'].value_counts()))

print 'fln counter'
for i in range(len(indexes)):
    single_vis = testset.loc[indexes[i]]
    bought = np.array(single_vis['ScanCount'] > 0)
    returned = np.array(single_vis['ScanCount'] < 0)
    single_vis_bought = single_vis.iloc[bought]
    single_vis_returned = single_vis.iloc[returned]

    if single_vis_bought.shape[0] == 0:
        test_fln_count_b.loc[indexes[i]] = 0
    else:
        if len(single_vis_bought.shape) == 1:
            test_fln_count_b.loc[indexes[i]] = 1
        else:
            test_fln_count_b.loc[indexes[i]] = len(list(single_vis_bought['FinelineNumber'].value_counts()))

    if single_vis_returned.shape[0] == 0:
        test_fln_count_r.loc[indexes[i]] = 0
    else:
        if len(single_vis_returned.shape) == 1:
            test_fln_count_r.loc[indexes[i]] = 1
        else:
            test_fln_count_r.loc[indexes[i]] = len(list(single_vis_returned['FinelineNumber'].value_counts()))

print 'upc counter'
for i in range(len(indexes)):
    single_vis = testset.loc[indexes[i]]
    bought = np.array(single_vis['ScanCount'] > 0)
    returned = np.array(single_vis['ScanCount'] < 0)
    single_vis_bought = single_vis.iloc[bought]
    single_vis_returned = single_vis.iloc[returned]

    if single_vis_bought.shape[0] == 0:
        test_upc_count_b.loc[indexes[i]] = 0
    else:
        if len(single_vis_bought.shape) == 1:
            test_upc_count_b.loc[indexes[i]] = 1
        else:
            test_upc_count_b.loc[indexes[i]] = len(list(single_vis_bought['Upc'].value_counts()))

    if single_vis_returned.shape[0] == 0:
        test_upc_count_r.loc[indexes[i]] = 0
    else:
        if len(single_vis_returned.shape) == 1:
            test_upc_count_r.loc[indexes[i]] = 1
        else:
            test_upc_count_r.loc[indexes[i]] = len(list(single_vis_returned['Upc'].value_counts()))

test = pd.concat([test_data_not_count, test_data_count_dep_bought, test_data_count_dep_returned,
                  test_data_count_fln_bought, test_data_count_fln_returned, test_data_count_upc,
                  test_dep_count_b, test_dep_count_r, test_fln_count_b, test_fln_count_r,
                  test_upc_count_b, test_upc_count_r, test_data_bought_items, test_data_returned_items], axis=1)
# test = remove_sparse(test)

# Find common coloumns
col_train = list(train.columns.values)
print col_train
col_test = list(test.columns.values)
print col_test
col_common = []
# add only common columns for train and test
for col in col_train:
    if col in col_test:
        col_common.append(col)
train = train[col_common]
test = test[col_common]
print col_common

print 'write to data'
train.to_csv("train_dummied_500_sep_dep_fln_b_r_v2.csv")
test.to_csv("test_dummied_500_sep_dep_fln_b_r_v2.csv")
