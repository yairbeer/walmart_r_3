import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2

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

n_trips = train_result.shape[0]

train_data_not_count = pd.get_dummies(trainset['Weekday'])
train_data_not_count = train_data_not_count.groupby(by=train_data_not_count.index, sort=False).mean()

# separate between returned and bought goods
train_total_items = np.array(trainset['ScanCount']).reshape((n, 1))
train_bought_items = np.clip(train_total_items, a_min=0, a_max=99999)
train_returned_items = np.clip(train_total_items, a_min=-99999, a_max=0)

train_data_tot_items = trainset['ScanCount'].groupby(by=trainset.index, sort=False).sum()
train_bought_items = pd.DataFrame(train_bought_items.ravel())
train_bought_items.index = trainset.index
train_bought_items.columns = ['Bought']
train_data_bought_items = train_bought_items.groupby(by=trainset.index, sort=False).sum()
train_returned_items = pd.DataFrame(train_returned_items.ravel())
train_returned_items.index = trainset.index
train_returned_items.columns = ['Returned']
train_data_returned_items = train_returned_items.groupby(by=trainset.index, sort=False).sum()

sparsity = 500

# bought Upc engineered
parsed_series = np.array(trainset['Upc']).astype('str')


def parse_rule(string):
    return string[:2]
vec_parse_rule = np.vectorize(parse_rule)

parsed_series = vec_parse_rule(parsed_series)
parsed_series = pd.DataFrame(parsed_series)
parsed_series.columns = ['parsing']
parsed_series.index = train_result.index

parsed_density = parsed_series.value_counts()
n_features = np.sum(parsed_density > 0)
print n_features
upc_density = parsed_density.iloc[:sparsity]
upc_density = list(upc_density.index)

# remove sparse Upc
tmp_series = np.zeros((trainset.shape[0], 1))
for i in range(trainset.shape[0]):
    upc_number = parsed_series.iloc[i]['parsing']
    if upc_number in upc_density:
        tmp_series[i] = upc_number
parsed_series['parsing'] = tmp_series
print parsed_series['parsing'].value_counts()

# dummy Upc
print 'dummy train Upc'
train_data_count_upc = pd.get_dummies(parsed_series['parsing'])
tmp_index = train_data_count_upc.index
tmp_columns = list(train_data_count_upc.columns.values)
tmp_table = np.array(train_data_count_upc) * train_total_items
train_data_count_upc = pd.DataFrame(tmp_table)
train_data_count_upc.columns = tmp_columns
train_data_count_upc.index = tmp_index
train_data_count_upc = train_data_count_upc.groupby(by=train_data_count_upc.index, sort=False).sum()
train_data_count_upc = add_prefix(train_data_count_upc, 'Upc_')

train_upc_count_b = np.ones((train_result.shape[0], 1))
train_upc_count_b = pd.DataFrame(train_upc_count_b)
train_upc_count_b.index = train_data_count_upc.index
train_upc_count_r = np.ones((train_result.shape[0], 1))
train_upc_count_r = pd.DataFrame(train_upc_count_r)
train_upc_count_r.index = train_data_count_upc.index

train_upc_count_b.columns = ['upc_num_B']
train_upc_count_r.columns = ['upc_num_R']
indexes = list(train_data_count_upc.index.values)

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

print 'absing'
train_arr = np.array(train_data_count_upc)
for i in range(train_data_count_upc.shape[0]):
    for j in range(train_data_count_upc.shape[1]):
        train_arr[i, j] = np.abs(train_arr[i, j])

chi2_params = chi2(train_arr, train_result)