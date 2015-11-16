import pandas as pd
import numpy as np

__author__ = 'WiBeer'


def add_prefix(dataset, prefix):
    col_names = np.array(dataset.columns.values).astype('str')
    for i_in in range(col_names.shape[0]):
        col_names[i_in] = prefix + col_names[i_in]
    dataset.columns = col_names
    return dataset


def remove_sparse(dataset, n_valid_samples):
    col_names = list(dataset.columns.values)
    dead_cols = []
    for i_in in range(len(col_names)):
        if np.sum(dataset[col_names[i_in]]) < sparsity:
            dead_cols.append(col_names[i_in])
    dataset = dataset.drop(dead_cols)
    print dead_cols
    return dataset


def parse_rule(string, digit):
    return string[digit]
vec_parse_rule = np.vectorize(parse_rule)


def dummy_sep(df, col, bought, returned):
    dum = pd.get_dummies(df[col])
    tmp_index = dum.index
    tmp_columns = list(dum.columns.values)

    # separate b/r department variables
    # bought Department
    tmp_table = np.array(dum) * bought
    count_bought = pd.DataFrame(tmp_table)
    count_bought.columns = tmp_columns
    count_bought.index = tmp_index
    count_bought = count_bought.groupby(by=count_bought.index, sort=False).sum()
    count_bought = add_prefix(count_bought, col + '_B_')

    # returned Department
    tmp_table = np.array(dum) * returned
    count_returned = pd.DataFrame(tmp_table)
    count_returned.columns = tmp_columns
    count_returned.index = tmp_index
    count_returned = count_returned.groupby(by=count_returned.index, sort=False).sum()
    count_returned = add_prefix(count_returned, col + '_R_')
    return count_bought, count_returned


def dummy_sep_sparse(df, col, sparsity, bought, returned):
    col_density = df[col].value_counts()
    n_features = np.sum(col_density > sparsity)
    # print n_features
    col_density = col_density.iloc[:n_features]
    dummy_cols = list(col_density.index)

    # remove sparse FinelineNumber products
    tmp_series = np.zeros((df.shape[0], 1))
    for i in range(df.shape[0]):
        flnumber = df.iloc[i][col]
        if flnumber in fineline_cols:
            tmp_series[i] = flnumber
    df[col] = tmp_series
    print df[col].value_counts()

    # dummy fln
    print 'dummy train FinelineNumber'

    # bought
    data_count = pd.get_dummies(df[col])
    tmp_index = data_count.index
    tmp_columns = list(data_count.columns.values)
    tmp_table = np.array(data_count) * np.array(bought)
    count_bought = pd.DataFrame(tmp_table)
    count_bought.columns = tmp_columns
    count_bought.index = tmp_index
    count_bought = count_bought.groupby(by=count_bought.index, sort=False).sum()
    count_bought = add_prefix(count_bought, col + '_B_')

    # returned
    data_count = pd.get_dummies(df['FinelineNumber'])
    tmp_index = data_count.index
    tmp_columns = list(data_count.columns.values)
    tmp_table = np.array(data_count) * np.array(returned)
    count_returned = pd.DataFrame(tmp_table)
    count_returned.columns = tmp_columns
    count_returned.index = tmp_index
    count_returned = count_returned.groupby(by=count_returned.index, sort=False).sum()
    count_returned = add_prefix(count_returned, col + '_R_')
    return count_bought, count_returned


"""
preprocessing data
"""
# preprocess test data
print 'read train data'
trainset = pd.DataFrame.from_csv('train.csv', index_col=1)
trainset = trainset.fillna('9999')
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

train_dep_num_b = np.ones((train_result.shape[0], 1))
train_dep_num_b = pd.DataFrame(train_dep_num_b)
train_dep_num_b.index = train_result.index
train_dep_num_r = np.ones((train_result.shape[0], 1))
train_dep_num_r = pd.DataFrame(train_dep_num_r)
train_dep_num_r.index = train_result.index

train_fln_num_b = np.ones((train_result.shape[0], 1))
train_fln_num_b = pd.DataFrame(train_fln_num_b)
train_fln_num_b.index = train_result.index
train_fln_num_r = np.ones((train_result.shape[0], 1))
train_fln_num_r = pd.DataFrame(train_fln_num_r)
train_fln_num_r.index = train_result.index

train_upc_num_b = np.ones((train_result.shape[0], 1))
train_upc_num_b = pd.DataFrame(train_upc_num_b)
train_upc_num_b.index = train_result.index
train_upc_num_r = np.ones((train_result.shape[0], 1))
train_upc_num_r = pd.DataFrame(train_upc_num_r)
train_upc_num_r.index = train_result.index

train_dep_num_b.columns = ['dep_num_B']
train_dep_num_r.columns = ['dep_num_R']

train_fln_num_b.columns = ['fln_num_B']
train_fln_num_r.columns = ['fln_num_R']

train_upc_num_b.columns = ['upc_num_B']
train_upc_num_r.columns = ['upc_num_R']
indexes = list(train_result.index.values)

print 'Department counter'
for i in range(len(indexes)):
    single_vis = trainset.loc[indexes[i]]
    bought = np.array(single_vis['ScanCount'] > 0)
    returned = np.array(single_vis['ScanCount'] < 0)
    single_vis_bought = single_vis.iloc[bought]
    single_vis_returned = single_vis.iloc[returned]

    if single_vis_bought.shape[0] == 0:
        train_dep_num_b.loc[indexes[i]] = 0
    else:
        if len(single_vis_bought.shape) == 1:
            train_dep_num_b.loc[indexes[i]] = 1
        else:
            train_dep_num_b.loc[indexes[i]] = len(list(single_vis_bought['DepartmentDescription'].value_counts()))

    if single_vis_returned.shape[0] == 0:
        train_dep_num_r.loc[indexes[i]] = 0
    else:
        if len(single_vis_returned.shape) == 1:
            train_dep_num_r.loc[indexes[i]] = 1
        else:
            train_dep_num_r.loc[indexes[i]] = len(list(single_vis_returned['DepartmentDescription'].value_counts()))

print 'fln counter'
for i in range(len(indexes)):
    single_vis = trainset.loc[indexes[i]]
    bought = np.array(single_vis['ScanCount'] > 0)
    returned = np.array(single_vis['ScanCount'] < 0)
    single_vis_bought = single_vis.iloc[bought]
    single_vis_returned = single_vis.iloc[returned]

    if single_vis_bought.shape[0] == 0:
        train_fln_num_b.loc[indexes[i]] = 0
    else:
        if len(single_vis_bought.shape) == 1:
            train_fln_num_b.loc[indexes[i]] = 1
        else:
            train_fln_num_b.loc[indexes[i]] = len(list(single_vis_bought['FinelineNumber'].value_counts()))

    if single_vis_returned.shape[0] == 0:
        train_fln_num_r.loc[indexes[i]] = 0
    else:
        if len(single_vis_returned.shape) == 1:
            train_fln_num_r.loc[indexes[i]] = 1
        else:
            train_fln_num_r.loc[indexes[i]] = len(list(single_vis_returned['FinelineNumber'].value_counts()))

print 'upc counter'
for i in range(len(indexes)):
    single_vis = trainset.loc[indexes[i]]
    bought = np.array(single_vis['ScanCount'] > 0)
    returned = np.array(single_vis['ScanCount'] < 0)
    single_vis_bought = single_vis.iloc[bought]
    single_vis_returned = single_vis.iloc[returned]

    if single_vis_bought.shape[0] == 0:
        train_upc_num_b.loc[indexes[i]] = 0
    else:
        if len(single_vis_bought.shape) == 1:
            train_upc_num_b.loc[indexes[i]] = 1
        else:
            train_upc_num_b.loc[indexes[i]] = len(list(single_vis_bought['Upc'].value_counts()))

    if single_vis_returned.shape[0] == 0:
        train_upc_num_r.loc[indexes[i]] = 0
    else:
        if len(single_vis_returned.shape) == 1:
            train_upc_num_r.loc[indexes[i]] = 1
        else:
            train_upc_num_r.loc[indexes[i]] = len(list(single_vis_returned['Upc'].value_counts()))

print 'dummy train DepartmentDescription'
train_count_dep_bought, train_count_dep_returned = dummy_sep(trainset, 'DepartmentDescription')

sparsity = 1000

# # bought Fln engineered
# parsed_series = np.array(trainset['FinelineNumber']).astype('str')
# parsed_series = vec_parse_rule(parsed_series, 2)
# parsed_series = pd.DataFrame(parsed_series)
# parsed_series.columns = ['fln_subcat_2']
# parsed_series.index = trainset.index
# # print parsed_series
#
# # print parsed_series
# parsed_density = parsed_series['fln_subcat_2'].value_counts()
# # print parsed_density
#
# n_features = np.sum(parsed_density > sparsity)
# print n_features
#
# fln_density = parsed_density.iloc[:n_features]
# fln_density = list(fln_density.index)
#
# # remove sparse Upc
# tmp_series = np.zeros((trainset.shape[0], 1))
# for i in range(trainset.shape[0]):
#     fln_number = parsed_series.iloc[i]['fln_subcat_2']
#     if fln_number in fln_density:
#         tmp_series[i] = fln_number
# parsed_series['fln_subcat_2'] = tmp_series
# # print parsed_series['upc_subcat'].value_counts()
#
# # dummy sub Upc
# print 'dummy train sub Fln'
# train_data_count_fln_sub_2 = pd.get_dummies(parsed_series['fln_subcat_2'])
# tmp_index = train_data_count_fln_sub_2.index
# tmp_columns = list(train_data_count_fln_sub_2.columns.values)
# tmp_table = np.array(train_data_count_fln_sub_2) * train_total_items
# train_data_count_fln_sub_2 = pd.DataFrame(tmp_table)
# train_data_count_fln_sub_2.columns = tmp_columns
# train_data_count_fln_sub_2.index = tmp_index
# train_data_count_fln_sub_2 = train_data_count_fln_sub_2.groupby(by=train_data_count_fln_sub_2.index, sort=False).sum()
# train_data_count_fln_sub_2 = add_prefix(train_data_count_fln_sub_2, 'fln_subcat_2')
#
# # bought Upc engineered
# parsed_series = np.array(trainset['FinelineNumber']).astype('str')
# parsed_series = vec_parse_rule(parsed_series, 3)
# parsed_series = pd.DataFrame(parsed_series)
# parsed_series.columns = ['fln_subcat_3']
# parsed_series.index = trainset.index
# # print parsed_series
#
# # print parsed_series
# parsed_density = parsed_series['fln_subcat_3'].value_counts()
# # print parsed_density
#
# n_features = np.sum(parsed_density > sparsity)
# print n_features
#
# fln_density = parsed_density.iloc[:n_features]
# fln_density = list(fln_density.index)
#
# # remove sparse Upc
# tmp_series = np.zeros((trainset.shape[0], 1))
# for i in range(trainset.shape[0]):
#     fln_number = parsed_series.iloc[i]['fln_subcat_3']
#     if fln_number in fln_density:
#         tmp_series[i] = fln_number
# parsed_series['fln_subcat_3'] = tmp_series
# # print parsed_series['upc_subcat'].value_counts()
#
# # dummy sub Upc
# print 'dummy train sub Fln'
# train_data_count_fln_sub_3 = pd.get_dummies(parsed_series['fln_subcat_3'])
# tmp_index = train_data_count_fln_sub_3.index
# tmp_columns = list(train_data_count_fln_sub_3.columns.values)
# tmp_table = np.array(train_data_count_fln_sub_3) * train_total_items
# train_data_count_fln_sub_3 = pd.DataFrame(tmp_table)
# train_data_count_fln_sub_3.columns = tmp_columns
# train_data_count_fln_sub_3.index = tmp_index
# train_data_count_fln_sub_3 = train_data_count_fln_sub_3.groupby(by=train_data_count_fln_sub_3.index, sort=False).sum()
# train_data_count_fln_sub_3 = add_prefix(train_data_count_fln_sub_3, 'fln_subcat_3')

# find most bought FinelineNumber
print 'dummy train FinelineNumber'
train_count_fln_bought, train_count_fln_returned = dummy_sep_sparse(trainset, 'FinelineNumber', sparsity,
                                                                              train_bought_items, train_returned_items)

print 'dummy train Upc'
train_count_upc_bought, train_count_upc_returned = dummy_sep_sparse(trainset, 'Upc', sparsity,
                                                                              train_bought_items, train_returned_items)

train = pd.concat([train_data_not_count, train_count_dep_bought, train_count_dep_returned,
                   train_count_fln_bought, train_count_fln_returned,
                   train_count_upc_bought, train_count_upc_returned,
                   train_dep_num_b, train_dep_num_r,
                   train_fln_num_b, train_fln_num_r,
                   train_upc_num_b, train_upc_num_r,
                   train_bought_items, train_returned_items], axis=1)
train = remove_sparse(train, 1000)

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

# bought Fln engineered
parsed_series = np.array(testset['FinelineNumber']).astype('str')
parsed_series = vec_parse_rule(parsed_series, 2)
parsed_series = pd.DataFrame(parsed_series)
parsed_series.columns = ['fln_subcat_2']
parsed_series.index = testset.index

# print parsed_series

parsed_density = parsed_series['fln_subcat_2'].value_counts()
print parsed_series

n_features = np.sum(parsed_density > sparsity)
print n_features

fln_density = parsed_density.iloc[:n_features]
fln_density = list(fln_density.index)

# remove sparse Upc
tmp_series = np.zeros((testset.shape[0], 1))
for i in range(testset.shape[0]):
    fln_number = parsed_series.iloc[i]['fln_subcat_2']
    if fln_number in upc_density:
        tmp_series[i] = fln_density
parsed_series['fln_subcat_2'] = tmp_series
print parsed_series['fln_subcat_2'].value_counts()

# dummy Upc
print 'dummy test sub Fln'
test_data_count_fln_sub_2 = pd.get_dummies(parsed_series['fln_subcat_2'])
tmp_index = test_data_count_fln_sub_2.index
tmp_columns = list(test_data_count_fln_sub_2.columns.values)
tmp_table = np.array(test_data_count_fln_sub_2) * test_total_items
test_data_count_fln_sub_2 = pd.DataFrame(tmp_table)
test_data_count_fln_sub_2.columns = tmp_columns
test_data_count_fln_sub_2.index = tmp_index
test_data_count_fln_sub_2 = test_data_count_fln_sub_2.groupby(by=test_data_count_fln_sub_2.index, sort=False).sum()
test_data_count_fln_sub_2 = add_prefix(test_data_count_fln_sub_2, 'fln_subcat_2')

# bought Fln engineered
parsed_series = np.array(testset['FinelineNumber']).astype('str')
parsed_series = vec_parse_rule(parsed_series, 3)
parsed_series = pd.DataFrame(parsed_series)
parsed_series.columns = ['fln_subcat_3']
parsed_series.index = testset.index

# print parsed_series

parsed_density = parsed_series['fln_subcat_3'].value_counts()
print parsed_series

n_features = np.sum(parsed_density > sparsity)
print n_features

fln_density = parsed_density.iloc[:n_features]
fln_density = list(fln_density.index)

# remove sparse Upc
tmp_series = np.zeros((testset.shape[0], 1))
for i in range(testset.shape[0]):
    fln_number = parsed_series.iloc[i]['fln_subcat_3']
    if fln_number in fln_density:
        tmp_series[i] = upc_number
parsed_series['fln_subcat_3'] = tmp_series
print parsed_series['fln_subcat_3'].value_counts()

# dummy Upc
print 'dummy test sub Fln'
test_data_count_fln_sub_3 = pd.get_dummies(parsed_series['fln_subcat_3'])
tmp_index = test_data_count_fln_sub_3.index
tmp_columns = list(test_data_count_fln_sub_3.columns.values)
tmp_table = np.array(test_data_count_fln_sub_3) * test_total_items
test_data_count_fln_sub_3 = pd.DataFrame(tmp_table)
test_data_count_fln_sub_3.columns = tmp_columns
test_data_count_fln_sub_3.index = tmp_index
test_data_count_fln_sub_3 = test_data_count_fln_sub_3.groupby(by=test_data_count_fln_sub_3.index, sort=False).sum()
test_data_count_fln_sub_3 = add_prefix(test_data_count_fln_sub_3, 'fln_subcat_3')

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

# # returned
# test_data_count_fln = pd.get_dummies(testset['FinelineNumber'])
# tmp_index = test_data_count_fln.index
# tmp_columns = list(test_data_count_fln.columns.values)
# tmp_table = np.array(test_data_count_fln) * np.array(test_total_items)
# test_data_count_fln_returned = pd.DataFrame(tmp_table)
# test_data_count_fln_returned.columns = tmp_columns
# test_data_count_fln_returned.index = tmp_index
# test_data_count_fln_returned = test_data_count_fln_returned.groupby(by=test_data_count_fln_returned.index, sort=False).sum()
# test_data_count_fln_returned = add_prefix(test_data_count_fln_returned, 'FLN_R_')

# bought Upc engineered
parsed_series = np.array(testset['Upc']).astype('str')
parsed_series = vec_parse_rule(parsed_series, 2)
parsed_series = pd.DataFrame(parsed_series)
parsed_series.columns = ['upc_subcat_2']
parsed_series.index = testset.index

# print parsed_series

parsed_density = parsed_series['upc_subcat_2'].value_counts()
print parsed_series

n_features = np.sum(parsed_density > sparsity)
print n_features

upc_density = parsed_density.iloc[:n_features]
upc_density = list(upc_density.index)

# remove sparse Upc
tmp_series = np.zeros((testset.shape[0], 1))
for i in range(testset.shape[0]):
    upc_number = parsed_series.iloc[i]['upc_subcat_2']
    if upc_number in upc_density:
        tmp_series[i] = upc_number
parsed_series['upc_subcat_2'] = tmp_series
print parsed_series['upc_subcat_2'].value_counts()

# dummy Upc
print 'dummy test sub Upc'
test_data_count_upc_sub_2 = pd.get_dummies(parsed_series['upc_subcat_2'])
tmp_index = test_data_count_upc_sub_2.index
tmp_columns = list(test_data_count_upc_sub_2.columns.values)
tmp_table = np.array(test_data_count_upc_sub_2) * test_total_items
test_data_count_upc_sub_2 = pd.DataFrame(tmp_table)
test_data_count_upc_sub_2.columns = tmp_columns
test_data_count_upc_sub_2.index = tmp_index
test_data_count_upc_sub_2 = test_data_count_upc_sub_2.groupby(by=test_data_count_upc_sub_2.index, sort=False).sum()
test_data_count_upc_sub_2 = add_prefix(test_data_count_upc_sub_2, 'upc_subcat_2')

# bought Upc engineered
parsed_series = np.array(testset['Upc']).astype('str')
parsed_series = vec_parse_rule(parsed_series, 3)
parsed_series = pd.DataFrame(parsed_series)
parsed_series.columns = ['upc_subcat_3']
parsed_series.index = testset.index

# print parsed_series

parsed_density = parsed_series['upc_subcat_3'].value_counts()
print parsed_series

n_features = np.sum(parsed_density > sparsity)
print n_features

upc_density = parsed_density.iloc[:n_features]
upc_density = list(upc_density.index)

# remove sparse Upc
tmp_series = np.zeros((testset.shape[0], 1))
for i in range(testset.shape[0]):
    upc_number = parsed_series.iloc[i]['upc_subcat_3']
    if upc_number in upc_density:
        tmp_series[i] = upc_number
parsed_series['upc_subcat_3'] = tmp_series
print parsed_series['upc_subcat_3'].value_counts()

# dummy Upc
print 'dummy test sub Upc'
test_data_count_upc_sub_3 = pd.get_dummies(parsed_series['upc_subcat_3'])
tmp_index = test_data_count_upc_sub_3.index
tmp_columns = list(test_data_count_upc_sub_3.columns.values)
tmp_table = np.array(test_data_count_upc_sub_3) * test_total_items
test_data_count_upc_sub_3 = pd.DataFrame(tmp_table)
test_data_count_upc_sub_3.columns = tmp_columns
test_data_count_upc_sub_3.index = tmp_index
test_data_count_upc_sub_3 = test_data_count_upc_sub_3.groupby(by=test_data_count_upc_sub_3.index, sort=False).sum()
test_data_count_upc_sub_3 = add_prefix(test_data_count_upc_sub_3, 'upc_subcat_3')

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
                  test_data_count_fln_bought, test_data_count_fln_sub_2, test_data_count_fln_sub_3,
                  # test_data_count_fln_returned,
                  test_data_count_upc_sub_2, test_data_count_upc_sub_3,
                  test_data_count_upc, test_dep_count_b, test_dep_count_r, test_fln_count_b, test_fln_count_r,
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
train.to_csv("train_dummied_1000_sep_dep_fln_b_r_v5.csv")
test.to_csv("test_dummied_1000_sep_dep_fln_b_r_v5.csv")
