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
# print train_result.value_counts()
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

sparsity = 750

# bought Upc engineered
parsed_series = np.array(trainset['Upc']).astype('str')


def parse_rule(string):
    return string[:3]
vec_parse_rule = np.vectorize(parse_rule)

parsed_series = vec_parse_rule(parsed_series)
parsed_series = pd.DataFrame(parsed_series)
parsed_series.columns = ['parsing']
parsed_series.index = trainset.index

# print parsed_series

parsed_density = parsed_series['parsing'].value_counts()
print parsed_series

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

# train_upc_count_b = np.ones((train_result.shape[0], 1))
# train_upc_count_b = pd.DataFrame(train_upc_count_b)
# train_upc_count_b.index = train_data_count_upc.index
# train_upc_count_r = np.ones((train_result.shape[0], 1))
# train_upc_count_r = pd.DataFrame(train_upc_count_r)
# train_upc_count_r.index = train_data_count_upc.index
#
# train_upc_count_b.columns = ['upc_num_B']
# train_upc_count_r.columns = ['upc_num_R']
# indexes = list(train_data_count_upc.index.values)
#
# print 'upc counter'
# for i in range(len(indexes)):
#     single_vis = trainset.loc[indexes[i]]
#     bought = np.array(single_vis['ScanCount'] > 0)
#     returned = np.array(single_vis['ScanCount'] < 0)
#     single_vis_bought = single_vis.iloc[bought]
#     single_vis_returned = single_vis.iloc[returned]
#
#     if single_vis_bought.shape[0] == 0:
#         train_upc_count_b.loc[indexes[i]] = 0
#     else:
#         if len(single_vis_bought.shape) == 1:
#             train_upc_count_b.loc[indexes[i]] = 1
#         else:
#             train_upc_count_b.loc[indexes[i]] = len(list(single_vis_bought['Upc'].value_counts()))
#
#     if single_vis_returned.shape[0] == 0:
#         train_upc_count_r.loc[indexes[i]] = 0
#     else:
#         if len(single_vis_returned.shape) == 1:
#             train_upc_count_r.loc[indexes[i]] = 1
#         else:
#             train_upc_count_r.loc[indexes[i]] = len(list(single_vis_returned['Upc'].value_counts()))
#
# train = pd.concat([train_upc_count_b, train_upc_count_r], axis=1)

col_list = list(train_data_count_upc.columns.values)

print 'absing'
train_arr = np.array(train_data_count_upc)
for i in range(train_data_count_upc.shape[0]):
    for j in range(train_data_count_upc.shape[1]):
        train_arr[i, j] = np.abs(train_arr[i, j])

chi2_params = chi2(train_arr, train_result)

for i in range(train_arr.shape[1]):
    print col_list[i], chi2_params[0][i]

"""
2 digits chi2
"""
# Upc_-9.0 30698.4212134
# Upc_3.0 740.323260848
# Upc_4.0 4498.1015281
# Upc_5.0 1212.81103699
# Upc_6.0 100028.736862
# Upc_7.0 521.122991591
# Upc_8.0 20360.5502769
# Upc_9.0 2504.05784375
# Upc_10.0 2787.84944309
# Upc_11.0 9983.43236478
# Upc_12.0 8217.31703759
# Upc_13.0 12071.9698733
# Upc_14.0 10582.9950412
# Upc_15.0 40344.0482936
# Upc_16.0 13266.8564203
# Upc_17.0 7297.07479859
# Upc_18.0 7513.62245018
# Upc_19.0 6647.78943276
# Upc_20.0 18371.0396851
# Upc_21.0 21108.3279966
# Upc_22.0 20720.6472418
# Upc_23.0 12760.2662788
# Upc_24.0 28244.8123507
# Upc_25.0 13973.9063182
# Upc_26.0 4006.05258918
# Upc_27.0 14787.0701536
# Upc_28.0 23378.333816
# Upc_29.0 6271.72860778
# Upc_30.0 15101.538768
# Upc_31.0 9850.10191436
# Upc_32.0 3784.85267542
# Upc_33.0 32124.3136048
# Upc_34.0 9601.50154265
# Upc_35.0 12246.7224145
# Upc_36.0 14657.6166976
# Upc_37.0 33565.7997358
# Upc_38.0 18970.117766
# Upc_39.0 6583.4045619
# Upc_40.0 63231.7493991
# Upc_41.0 50542.3001455
# Upc_42.0 6777.99241931
# Upc_43.0 13551.3265846
# Upc_44.0 30758.9780617
# Upc_45.0 2026.40281601
# Upc_46.0 16920.3867013
# Upc_47.0 3983.44566681
# Upc_48.0 12341.3292265
# Upc_49.0 20940.3347504
# Upc_50.0 33525.0237266
# Upc_51.0 15122.0167203
# Upc_52.0 19069.7588766
# Upc_53.0 1638.68272071
# Upc_54.0 5755.8464442
# Upc_55.0 894.013820134
# Upc_56.0 1722.58802749
# Upc_57.0 243.412155504
# Upc_58.0 700.768174012
# Upc_59.0 248.324152191
# Upc_60.0 20476.7121445
# Upc_61.0 5284.16281237
# Upc_62.0 3596.2004697
# Upc_63.0 3287.43735188
# Upc_64.0 19629.4486732
# Upc_65.0 4865.41876489
# Upc_66.0 4971.55562636
# Upc_67.0 3172.03419518
# Upc_68.0 19987.5428833
# Upc_69.0 4271.15734871
# Upc_70.0 24030.5148424
# Upc_71.0 23657.6152219
# Upc_72.0 20591.3130762
# Upc_73.0 14674.6873185
# Upc_74.0 15665.3665567
# Upc_75.0 12841.962101
# Upc_76.0 14384.7557602
# Upc_77.0 9183.77886797
# Upc_78.0 126338.644381
# Upc_79.0 12629.2431433
# Upc_80.0 8563.58637256
# Upc_81.0 8747.36818275
# Upc_82.0 5941.65485555
# Upc_83.0 2570.47920293
# Upc_84.0 7672.38384699
# Upc_85.0 5336.44363743
# Upc_86.0 2610.85237746
# Upc_87.0 3584.96740384
# Upc_88.0 32530.7551696
# Upc_89.0 4888.83936239
# Upc_90.0 1738.98409709
# Upc_91.0 1938.93930347
# Upc_92.0 630.345504439
# Upc_93.0 1599.22668884
# Upc_94.0 606.67436271
# Upc_95.0 2812.04787394
# Upc_96.0 438.918994452
# Upc_97.0 1068.8163589
# Upc_98.0 1573.55242422
# Upc_99.0 936.388519657
"""
3 digits
"""
