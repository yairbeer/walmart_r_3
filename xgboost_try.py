from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.feature_selection import chi2
import xgboostlib.xgboost as xgboost

__author__ = 'YBeer'

train_result = pd.DataFrame.from_csv("train_result.csv")
col = list(train_result.columns.values)
result_ind = list(train_result[col[0]].value_counts().index)
# train_result = pd.get_dummies(train_result)
train_result = np.array(train_result).ravel()
train_result_xgb = np.zeros(train_result.shape)
for i in range(1, len(result_ind)):
    train_result_xgb += (train_result == result_ind[i]) * i
# print train_result_xgb

train = pd.DataFrame.from_csv("train_dummied_200_sep_dep_fln_b_r_v5.csv")
train.fillna(0)
train_arr = np.array(train)
col_list = list(train.columns.values)

# print train_result.shape[1], ' categorial'
print train.shape[1], ' columns'

print 'absing'
for i in range(train_arr.shape[0]):
    for j in range(train_arr.shape[1]):
        train_arr[i, j] = np.abs(train_arr[i, j])

chi2_params = chi2(train_arr, train_result)

# for i in range(train_arr.shape[1]):
#     print col_list[i], chi2_params[0][i]

del train_arr

best_metric = 10
best_params = []
param_grid = {'silent': [1], 'nthread': [4], 'num_class': [38], 'eval_metric': ['mlogloss'], 'eta': [0.1],
              'objective': ['multi:softprob'], 'max_depth': [5], 'chi2_lim': [0], 'num_round': [500]}

for params in ParameterGrid(param_grid):
    print params

    # filtering low chi2 cols
    chi2_lim = params['chi2_lim']
    chi2_cols = []
    for i in range(train.shape[1]):
        if chi2_params[0][i] > chi2_lim:
            chi2_cols.append(col_list[i])

    print len(chi2_cols), ' chi2 columns'
    train_arr = train.copy(deep=True)
    train_arr = train_arr[chi2_cols]

    # Standardizing
    stding = StandardScaler()
    train_arr = stding.fit_transform(train_arr)

    print 'start CV'

    # CV
    cv_n = 2
    X_train, X_test, y_train, y_test = train_test_split(train_arr, train_result_xgb, test_size=0.5, random_state=1)
    metric = []

    # train machine learning
    xg_train = xgboost.DMatrix(X_train, label=y_train)
    xg_test = xgboost.DMatrix(X_test, label=y_test)

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]

    num_round = params['num_round']
    xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);

    # predict
    class_pred = xgclassifier.predict(xg_test)
    class_pred = class_pred.reshape(y_test.shape[0], 38)

    # evaluate
    # print log_loss(y_test, class_pred)
    metric = log_loss(y_test, class_pred)

    print 'The log loss is: ', metric
    if metric < best_metric:
        best_metric = metric
        best_params = params
    print 'The best metric is:', best_metric, 'for the params:', best_params

# The best metric is: 0.788417796191 for the params:  {'num_class': 38, 'silent': 1, 'eval_metric': 'mlogloss', 'nthread': 4, 'objective': 'multi:softprob', 'eta': 0.1, 'num_round': 100, 'max_depth': 7, 'chi2_lim': 250}
# The best metric is:  0.761735967063 for the params:  {'num_class': 38, 'silent': 1, 'eval_metric': 'mlogloss', 'nthread': 4, 'objective': 'multi:softprob', 'eta': 0.1, 'num_round': 200, 'max_depth': 7, 'chi2_lim': 0}
# train-mlogloss:0.339249

# {'num_class': 38, 'silent': 1, 'eval_metric': 'mlogloss', 'nthread': 4, 'objective': 'multi:softprob', 'eta': 0.1, 'num_round': 500, 'max_depth': 5, 'chi2_lim': 0}
# 1197  chi2 columns
# [0]	train-mlogloss:2.791107	test-mlogloss:2.815224
# [1]	train-mlogloss:2.483790	test-mlogloss:2.519392
# [2]	train-mlogloss:2.269790	test-mlogloss:2.313475
# [3]	train-mlogloss:2.103512	test-mlogloss:2.154516
# [4]	train-mlogloss:1.967409	test-mlogloss:2.024796
# [5]	train-mlogloss:1.852296	test-mlogloss:1.915044
# [6]	train-mlogloss:1.752856	test-mlogloss:1.820625
# [7]	train-mlogloss:1.666849	test-mlogloss:1.739699
# [8]	train-mlogloss:1.590817	test-mlogloss:1.668175
# [9]	train-mlogloss:1.521220	test-mlogloss:1.603454
# [10]	train-mlogloss:1.460056	test-mlogloss:1.546551
# [11]	train-mlogloss:1.404379	test-mlogloss:1.495100
# [12]	train-mlogloss:1.353934	test-mlogloss:1.448590
# [13]	train-mlogloss:1.307886	test-mlogloss:1.406155
# [14]	train-mlogloss:1.265617	test-mlogloss:1.367847
# [15]	train-mlogloss:1.226923	test-mlogloss:1.332350
# [16]	train-mlogloss:1.191891	test-mlogloss:1.300631
# [17]	train-mlogloss:1.158544	test-mlogloss:1.270433
# [18]	train-mlogloss:1.127580	test-mlogloss:1.242791
# [19]	train-mlogloss:1.099357	test-mlogloss:1.217948
# [20]	train-mlogloss:1.073590	test-mlogloss:1.195257
# [21]	train-mlogloss:1.048995	test-mlogloss:1.173599
# [22]	train-mlogloss:1.025839	test-mlogloss:1.153255
# [23]	train-mlogloss:1.004223	test-mlogloss:1.134575
# [24]	train-mlogloss:0.984108	test-mlogloss:1.117076
# [25]	train-mlogloss:0.965116	test-mlogloss:1.100746
# [26]	train-mlogloss:0.946970	test-mlogloss:1.085250
# [27]	train-mlogloss:0.930279	test-mlogloss:1.071266
# [28]	train-mlogloss:0.914369	test-mlogloss:1.057846
# [29]	train-mlogloss:0.899613	test-mlogloss:1.045643
# [30]	train-mlogloss:0.885392	test-mlogloss:1.033707
# [31]	train-mlogloss:0.871601	test-mlogloss:1.022575
# [32]	train-mlogloss:0.858283	test-mlogloss:1.012144
# [33]	train-mlogloss:0.846046	test-mlogloss:1.002315
# [34]	train-mlogloss:0.834300	test-mlogloss:0.993152
# [35]	train-mlogloss:0.823117	test-mlogloss:0.984282
# [36]	train-mlogloss:0.812279	test-mlogloss:0.975968
# [37]	train-mlogloss:0.801793	test-mlogloss:0.967970
# [38]	train-mlogloss:0.791699	test-mlogloss:0.960369
# [39]	train-mlogloss:0.782576	test-mlogloss:0.953320
# [40]	train-mlogloss:0.773248	test-mlogloss:0.946407
# [41]	train-mlogloss:0.764146	test-mlogloss:0.939665
# [42]	train-mlogloss:0.755487	test-mlogloss:0.933297
# [43]	train-mlogloss:0.747412	test-mlogloss:0.927591
# [44]	train-mlogloss:0.739268	test-mlogloss:0.921879
# [45]	train-mlogloss:0.731670	test-mlogloss:0.916573
# [46]	train-mlogloss:0.724434	test-mlogloss:0.911580
# [47]	train-mlogloss:0.716923	test-mlogloss:0.906296
# [48]	train-mlogloss:0.709970	test-mlogloss:0.901268
# [49]	train-mlogloss:0.703361	test-mlogloss:0.896922
# [50]	train-mlogloss:0.696642	test-mlogloss:0.892561
# [51]	train-mlogloss:0.690372	test-mlogloss:0.888404
# [52]	train-mlogloss:0.684238	test-mlogloss:0.884357
# [53]	train-mlogloss:0.678387	test-mlogloss:0.880711
# [54]	train-mlogloss:0.672546	test-mlogloss:0.877009
# [55]	train-mlogloss:0.667101	test-mlogloss:0.873604
# [56]	train-mlogloss:0.661567	test-mlogloss:0.870158
# [57]	train-mlogloss:0.656395	test-mlogloss:0.866886
# [58]	train-mlogloss:0.650936	test-mlogloss:0.863601
# [59]	train-mlogloss:0.645731	test-mlogloss:0.860521
# [60]	train-mlogloss:0.641067	test-mlogloss:0.857825
# [61]	train-mlogloss:0.636689	test-mlogloss:0.855212
# [62]	train-mlogloss:0.631955	test-mlogloss:0.852462
# [63]	train-mlogloss:0.627317	test-mlogloss:0.849890
# [64]	train-mlogloss:0.623197	test-mlogloss:0.847485
# [65]	train-mlogloss:0.619013	test-mlogloss:0.845157
# [66]	train-mlogloss:0.614593	test-mlogloss:0.842785
# [67]	train-mlogloss:0.610316	test-mlogloss:0.840575
# [68]	train-mlogloss:0.606076	test-mlogloss:0.838351
# [69]	train-mlogloss:0.602006	test-mlogloss:0.836273
# [70]	train-mlogloss:0.598221	test-mlogloss:0.834418
# [71]	train-mlogloss:0.594119	test-mlogloss:0.832233
# [72]	train-mlogloss:0.590472	test-mlogloss:0.830422
# [73]	train-mlogloss:0.586576	test-mlogloss:0.828604
# [74]	train-mlogloss:0.582959	test-mlogloss:0.826804
# [75]	train-mlogloss:0.579271	test-mlogloss:0.825030
# [76]	train-mlogloss:0.575794	test-mlogloss:0.823361
# [77]	train-mlogloss:0.572162	test-mlogloss:0.821768
# [78]	train-mlogloss:0.568898	test-mlogloss:0.820422
# [79]	train-mlogloss:0.565425	test-mlogloss:0.818769
# [80]	train-mlogloss:0.562241	test-mlogloss:0.817483
# [81]	train-mlogloss:0.559163	test-mlogloss:0.816246
# [82]	train-mlogloss:0.555731	test-mlogloss:0.814875
# [83]	train-mlogloss:0.552782	test-mlogloss:0.813566
# [84]	train-mlogloss:0.549978	test-mlogloss:0.812391
# [85]	train-mlogloss:0.546972	test-mlogloss:0.811154
# [86]	train-mlogloss:0.544061	test-mlogloss:0.810011
# [87]	train-mlogloss:0.541224	test-mlogloss:0.808767
# [88]	train-mlogloss:0.538423	test-mlogloss:0.807583
# [89]	train-mlogloss:0.535605	test-mlogloss:0.806535
# [90]	train-mlogloss:0.532598	test-mlogloss:0.805481
# [91]	train-mlogloss:0.529979	test-mlogloss:0.804523
# [92]	train-mlogloss:0.527294	test-mlogloss:0.803371
# [93]	train-mlogloss:0.524515	test-mlogloss:0.802327
# [94]	train-mlogloss:0.521692	test-mlogloss:0.801312
# [95]	train-mlogloss:0.519039	test-mlogloss:0.800504
# [96]	train-mlogloss:0.516205	test-mlogloss:0.799512
# [97]	train-mlogloss:0.513727	test-mlogloss:0.798509
# [98]	train-mlogloss:0.510989	test-mlogloss:0.797562
# [99]	train-mlogloss:0.508501	test-mlogloss:0.796687
# [100]	train-mlogloss:0.506041	test-mlogloss:0.795766
# [101]	train-mlogloss:0.503680	test-mlogloss:0.794974
# [102]	train-mlogloss:0.501151	test-mlogloss:0.794028
# [103]	train-mlogloss:0.498832	test-mlogloss:0.793300
# [104]	train-mlogloss:0.496360	test-mlogloss:0.792405
# [105]	train-mlogloss:0.493859	test-mlogloss:0.791618
# [106]	train-mlogloss:0.491650	test-mlogloss:0.790967
# [107]	train-mlogloss:0.489237	test-mlogloss:0.790236
# [108]	train-mlogloss:0.486961	test-mlogloss:0.789422
# [109]	train-mlogloss:0.484730	test-mlogloss:0.788656
# [110]	train-mlogloss:0.482617	test-mlogloss:0.787949
# [111]	train-mlogloss:0.480314	test-mlogloss:0.787260
# [112]	train-mlogloss:0.478277	test-mlogloss:0.786673
# [113]	train-mlogloss:0.476039	test-mlogloss:0.786040
# [114]	train-mlogloss:0.474043	test-mlogloss:0.785547
# [115]	train-mlogloss:0.471882	test-mlogloss:0.784951
# [116]	train-mlogloss:0.469771	test-mlogloss:0.784430
# [117]	train-mlogloss:0.467806	test-mlogloss:0.783853
# [118]	train-mlogloss:0.465663	test-mlogloss:0.783210
# [119]	train-mlogloss:0.463661	test-mlogloss:0.782554
# [120]	train-mlogloss:0.461528	test-mlogloss:0.781990
# [121]	train-mlogloss:0.459525	test-mlogloss:0.781467
# [122]	train-mlogloss:0.457631	test-mlogloss:0.780950
# [123]	train-mlogloss:0.455648	test-mlogloss:0.780495
# [124]	train-mlogloss:0.453873	test-mlogloss:0.780016
# [125]	train-mlogloss:0.451999	test-mlogloss:0.779503
# [126]	train-mlogloss:0.450308	test-mlogloss:0.778958
# [127]	train-mlogloss:0.448406	test-mlogloss:0.778428
# [128]	train-mlogloss:0.446547	test-mlogloss:0.777907
# [129]	train-mlogloss:0.444630	test-mlogloss:0.777490
# [130]	train-mlogloss:0.442935	test-mlogloss:0.777042
# [131]	train-mlogloss:0.440978	test-mlogloss:0.776551
# [132]	train-mlogloss:0.439188	test-mlogloss:0.776117
# [133]	train-mlogloss:0.437400	test-mlogloss:0.775545
# [134]	train-mlogloss:0.435621	test-mlogloss:0.775073
# [135]	train-mlogloss:0.433932	test-mlogloss:0.774700
# [136]	train-mlogloss:0.432246	test-mlogloss:0.774330
# [137]	train-mlogloss:0.430424	test-mlogloss:0.773822
# [138]	train-mlogloss:0.428718	test-mlogloss:0.773381
# [139]	train-mlogloss:0.427133	test-mlogloss:0.773010
# [140]	train-mlogloss:0.425352	test-mlogloss:0.772584
# [141]	train-mlogloss:0.423575	test-mlogloss:0.772207
# [142]	train-mlogloss:0.422036	test-mlogloss:0.771935
# [143]	train-mlogloss:0.420364	test-mlogloss:0.771538
# [144]	train-mlogloss:0.418909	test-mlogloss:0.771260
# [145]	train-mlogloss:0.417311	test-mlogloss:0.770877
# [146]	train-mlogloss:0.415849	test-mlogloss:0.770521
# [147]	train-mlogloss:0.414023	test-mlogloss:0.770119
# [148]	train-mlogloss:0.412402	test-mlogloss:0.769664
# [149]	train-mlogloss:0.410770	test-mlogloss:0.769320
# [150]	train-mlogloss:0.409335	test-mlogloss:0.769046
# [151]	train-mlogloss:0.407834	test-mlogloss:0.768618
# [152]	train-mlogloss:0.406285	test-mlogloss:0.768225
# [153]	train-mlogloss:0.404672	test-mlogloss:0.767813
# [154]	train-mlogloss:0.403271	test-mlogloss:0.767555
# [155]	train-mlogloss:0.401811	test-mlogloss:0.767243
# [156]	train-mlogloss:0.400381	test-mlogloss:0.766903
# [157]	train-mlogloss:0.398906	test-mlogloss:0.766608
# [158]	train-mlogloss:0.397440	test-mlogloss:0.766252
# [159]	train-mlogloss:0.396032	test-mlogloss:0.765994
# [160]	train-mlogloss:0.394627	test-mlogloss:0.765742
# [161]	train-mlogloss:0.393412	test-mlogloss:0.765408
# [162]	train-mlogloss:0.391958	test-mlogloss:0.765066
# [163]	train-mlogloss:0.390477	test-mlogloss:0.764831
# [164]	train-mlogloss:0.388988	test-mlogloss:0.764495
# [165]	train-mlogloss:0.387771	test-mlogloss:0.764224
# [166]	train-mlogloss:0.386472	test-mlogloss:0.763963
# [167]	train-mlogloss:0.385196	test-mlogloss:0.763718
# [168]	train-mlogloss:0.383862	test-mlogloss:0.763463
# [169]	train-mlogloss:0.382512	test-mlogloss:0.763157
# [170]	train-mlogloss:0.381076	test-mlogloss:0.762994
# [171]	train-mlogloss:0.379834	test-mlogloss:0.762770
# [172]	train-mlogloss:0.378569	test-mlogloss:0.762479
# [173]	train-mlogloss:0.377233	test-mlogloss:0.762143
# [174]	train-mlogloss:0.375880	test-mlogloss:0.761889
# [175]	train-mlogloss:0.374685	test-mlogloss:0.761621
# [176]	train-mlogloss:0.373571	test-mlogloss:0.761425
# [177]	train-mlogloss:0.372414	test-mlogloss:0.761236
# [178]	train-mlogloss:0.371226	test-mlogloss:0.761026