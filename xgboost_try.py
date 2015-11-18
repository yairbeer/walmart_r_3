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

train = pd.DataFrame.from_csv("train_dummied_150_sep_dep_fln_b_r_v5.csv")
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
              'objective': ['multi:softprob'], 'max_depth': [4], 'chi2_lim': [250], 'num_round': [500]}

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

# C:\Users\yaia\Anaconda\python.exe C:/Users/yaia/Documents/GitHub/walmart_r_3/xgboost_try.py
# 1349  columns
# absing
# {'num_class': 38, 'silent': 1, 'eval_metric': 'mlogloss', 'nthread': 4, 'objective': 'multi:softprob', 'eta': 0.1, 'num_round': 500, 'max_depth': 4, 'chi2_lim': 250}
# 1315  chi2 columns
# C:\Users\yaia\Anaconda\lib\site-packages\sklearn\utils\validation.py:498: UserWarning: StandardScaler assumes floating point values as input, got int64
#   "got %s" % (estimator, X.dtype))
# start CV
# [0]	train-mlogloss:2.840051	test-mlogloss:2.857807
# [1]	train-mlogloss:2.541776	test-mlogloss:2.564883
# [2]	train-mlogloss:2.334474	test-mlogloss:2.362744
# [3]	train-mlogloss:2.173948	test-mlogloss:2.206222
# [4]	train-mlogloss:2.041281	test-mlogloss:2.076976
# [5]	train-mlogloss:1.930404	test-mlogloss:1.968570
# [6]	train-mlogloss:1.834987	test-mlogloss:1.876396
# [7]	train-mlogloss:1.750456	test-mlogloss:1.794552
# [8]	train-mlogloss:1.677308	test-mlogloss:1.724010
# [9]	train-mlogloss:1.611191	test-mlogloss:1.660691
# [10]	train-mlogloss:1.551803	test-mlogloss:1.603633
# [11]	train-mlogloss:1.498189	test-mlogloss:1.552069
# [12]	train-mlogloss:1.449524	test-mlogloss:1.505655
# [13]	train-mlogloss:1.404273	test-mlogloss:1.462853
# [14]	train-mlogloss:1.363179	test-mlogloss:1.424068
# [15]	train-mlogloss:1.325705	test-mlogloss:1.388621
# [16]	train-mlogloss:1.291190	test-mlogloss:1.356086
# [17]	train-mlogloss:1.259493	test-mlogloss:1.326197
# [18]	train-mlogloss:1.229740	test-mlogloss:1.298263
# [19]	train-mlogloss:1.202009	test-mlogloss:1.272472
# [20]	train-mlogloss:1.176468	test-mlogloss:1.248794
# [21]	train-mlogloss:1.152869	test-mlogloss:1.227195
# [22]	train-mlogloss:1.130597	test-mlogloss:1.206640
# [23]	train-mlogloss:1.109816	test-mlogloss:1.187669
# [24]	train-mlogloss:1.089921	test-mlogloss:1.169106
# [25]	train-mlogloss:1.071785	test-mlogloss:1.152695
# [26]	train-mlogloss:1.054624	test-mlogloss:1.137056
# [27]	train-mlogloss:1.038518	test-mlogloss:1.122484
# [28]	train-mlogloss:1.023176	test-mlogloss:1.108597
# [29]	train-mlogloss:1.008513	test-mlogloss:1.095604
# [30]	train-mlogloss:0.994743	test-mlogloss:1.083339
# [31]	train-mlogloss:0.981826	test-mlogloss:1.071946
# [32]	train-mlogloss:0.968788	test-mlogloss:1.060579
# [33]	train-mlogloss:0.956732	test-mlogloss:1.050060
# [34]	train-mlogloss:0.945486	test-mlogloss:1.040061
# [35]	train-mlogloss:0.934644	test-mlogloss:1.030828
# [36]	train-mlogloss:0.924082	test-mlogloss:1.021893
# [37]	train-mlogloss:0.914288	test-mlogloss:1.013591
# [38]	train-mlogloss:0.904774	test-mlogloss:1.005722
# [39]	train-mlogloss:0.895538	test-mlogloss:0.998003
# [40]	train-mlogloss:0.886721	test-mlogloss:0.990782
# [41]	train-mlogloss:0.878416	test-mlogloss:0.983855
# [42]	train-mlogloss:0.870415	test-mlogloss:0.977313
# [43]	train-mlogloss:0.862454	test-mlogloss:0.970766
# [44]	train-mlogloss:0.855299	test-mlogloss:0.964945
# [45]	train-mlogloss:0.848318	test-mlogloss:0.959323
# [46]	train-mlogloss:0.841248	test-mlogloss:0.953855
# [47]	train-mlogloss:0.834524	test-mlogloss:0.948685
# [48]	train-mlogloss:0.828309	test-mlogloss:0.943771
# [49]	train-mlogloss:0.821719	test-mlogloss:0.938739
# [50]	train-mlogloss:0.815619	test-mlogloss:0.934177
# [51]	train-mlogloss:0.809298	test-mlogloss:0.929311
# [52]	train-mlogloss:0.803690	test-mlogloss:0.925102
# [53]	train-mlogloss:0.798060	test-mlogloss:0.920897
# [54]	train-mlogloss:0.792740	test-mlogloss:0.916920
# [55]	train-mlogloss:0.787292	test-mlogloss:0.912868
# [56]	train-mlogloss:0.782221	test-mlogloss:0.909181
# [57]	train-mlogloss:0.777464	test-mlogloss:0.905834
# [58]	train-mlogloss:0.772590	test-mlogloss:0.902526
# [59]	train-mlogloss:0.767893	test-mlogloss:0.899192
# [60]	train-mlogloss:0.763449	test-mlogloss:0.896144
# [61]	train-mlogloss:0.758846	test-mlogloss:0.893100
# [62]	train-mlogloss:0.754407	test-mlogloss:0.890227
# [63]	train-mlogloss:0.750217	test-mlogloss:0.887418
# [64]	train-mlogloss:0.746140	test-mlogloss:0.884762
# [65]	train-mlogloss:0.742069	test-mlogloss:0.882212
# [66]	train-mlogloss:0.737961	test-mlogloss:0.879497
# [67]	train-mlogloss:0.734245	test-mlogloss:0.877140
# [68]	train-mlogloss:0.730512	test-mlogloss:0.874819
# [69]	train-mlogloss:0.726593	test-mlogloss:0.872290
# [70]	train-mlogloss:0.722897	test-mlogloss:0.869798
# [71]	train-mlogloss:0.719267	test-mlogloss:0.867514
# [72]	train-mlogloss:0.715427	test-mlogloss:0.865225
# [73]	train-mlogloss:0.712081	test-mlogloss:0.863148
# [74]	train-mlogloss:0.708791	test-mlogloss:0.861296
# [75]	train-mlogloss:0.705485	test-mlogloss:0.859399
# [76]	train-mlogloss:0.702492	test-mlogloss:0.857576
# [77]	train-mlogloss:0.699277	test-mlogloss:0.855724
# [78]	train-mlogloss:0.696081	test-mlogloss:0.853984
# [79]	train-mlogloss:0.692911	test-mlogloss:0.852165
# [80]	train-mlogloss:0.689906	test-mlogloss:0.850562
# [81]	train-mlogloss:0.687160	test-mlogloss:0.849157
# [82]	train-mlogloss:0.684019	test-mlogloss:0.847253
# [83]	train-mlogloss:0.681058	test-mlogloss:0.845694
# [84]	train-mlogloss:0.678197	test-mlogloss:0.844074
# [85]	train-mlogloss:0.675497	test-mlogloss:0.842693
# [86]	train-mlogloss:0.672720	test-mlogloss:0.841233
# [87]	train-mlogloss:0.670016	test-mlogloss:0.839725
# [88]	train-mlogloss:0.667197	test-mlogloss:0.838186
# [89]	train-mlogloss:0.664375	test-mlogloss:0.836789
# [90]	train-mlogloss:0.661521	test-mlogloss:0.835213
# [91]	train-mlogloss:0.658881	test-mlogloss:0.833781
# [92]	train-mlogloss:0.656372	test-mlogloss:0.832370
# [93]	train-mlogloss:0.653746	test-mlogloss:0.830946
# [94]	train-mlogloss:0.651181	test-mlogloss:0.829647
# [95]	train-mlogloss:0.648512	test-mlogloss:0.828429
# [96]	train-mlogloss:0.645992	test-mlogloss:0.827336
# [97]	train-mlogloss:0.643417	test-mlogloss:0.826233
# [98]	train-mlogloss:0.641067	test-mlogloss:0.825192
# [99]	train-mlogloss:0.638730	test-mlogloss:0.824039
# [100]	train-mlogloss:0.636165	test-mlogloss:0.822866
# [101]	train-mlogloss:0.633856	test-mlogloss:0.821731
# [102]	train-mlogloss:0.631650	test-mlogloss:0.820772
# [103]	train-mlogloss:0.629344	test-mlogloss:0.819770
# [104]	train-mlogloss:0.627198	test-mlogloss:0.818866
# [105]	train-mlogloss:0.624962	test-mlogloss:0.817824
# [106]	train-mlogloss:0.622985	test-mlogloss:0.816847
# [107]	train-mlogloss:0.620864	test-mlogloss:0.815954
# [108]	train-mlogloss:0.618683	test-mlogloss:0.814910
# [109]	train-mlogloss:0.616580	test-mlogloss:0.814027
# [110]	train-mlogloss:0.614351	test-mlogloss:0.813063
# [111]	train-mlogloss:0.612327	test-mlogloss:0.812280
# [112]	train-mlogloss:0.610129	test-mlogloss:0.811321
# [113]	train-mlogloss:0.608248	test-mlogloss:0.810633
# [114]	train-mlogloss:0.606120	test-mlogloss:0.809593
# [115]	train-mlogloss:0.604238	test-mlogloss:0.808823
# [116]	train-mlogloss:0.602133	test-mlogloss:0.807896
# [117]	train-mlogloss:0.600259	test-mlogloss:0.807131
# [118]	train-mlogloss:0.598300	test-mlogloss:0.806351
# [119]	train-mlogloss:0.596488	test-mlogloss:0.805649
# [120]	train-mlogloss:0.594497	test-mlogloss:0.804863
# [121]	train-mlogloss:0.592638	test-mlogloss:0.804082
# [122]	train-mlogloss:0.590715	test-mlogloss:0.803361
# [123]	train-mlogloss:0.589085	test-mlogloss:0.802739
# [124]	train-mlogloss:0.587154	test-mlogloss:0.801927
# [125]	train-mlogloss:0.585177	test-mlogloss:0.801150
# [126]	train-mlogloss:0.583237	test-mlogloss:0.800409
# [127]	train-mlogloss:0.581399	test-mlogloss:0.799717
# [128]	train-mlogloss:0.579777	test-mlogloss:0.799096
# [129]	train-mlogloss:0.578198	test-mlogloss:0.798507
# [130]	train-mlogloss:0.576594	test-mlogloss:0.797947
# [131]	train-mlogloss:0.574877	test-mlogloss:0.797272
# [132]	train-mlogloss:0.573171	test-mlogloss:0.796673
# [133]	train-mlogloss:0.571639	test-mlogloss:0.796103
# [134]	train-mlogloss:0.569671	test-mlogloss:0.795365
# [135]	train-mlogloss:0.567835	test-mlogloss:0.794718
# [136]	train-mlogloss:0.566176	test-mlogloss:0.794182
# [137]	train-mlogloss:0.564318	test-mlogloss:0.793452
# [138]	train-mlogloss:0.562680	test-mlogloss:0.792794
# [139]	train-mlogloss:0.561044	test-mlogloss:0.792223
# [140]	train-mlogloss:0.559476	test-mlogloss:0.791787
# [141]	train-mlogloss:0.557946	test-mlogloss:0.791202
# [142]	train-mlogloss:0.556431	test-mlogloss:0.790556
# [143]	train-mlogloss:0.554844	test-mlogloss:0.789867
# [144]	train-mlogloss:0.553365	test-mlogloss:0.789314
# [145]	train-mlogloss:0.551718	test-mlogloss:0.788702
# [146]	train-mlogloss:0.550269	test-mlogloss:0.788256
# [147]	train-mlogloss:0.548618	test-mlogloss:0.787692
# [148]	train-mlogloss:0.547345	test-mlogloss:0.787321
# [149]	train-mlogloss:0.545752	test-mlogloss:0.786720
# [150]	train-mlogloss:0.544284	test-mlogloss:0.786326
# [151]	train-mlogloss:0.542961	test-mlogloss:0.785792
# [152]	train-mlogloss:0.541562	test-mlogloss:0.785333
# [153]	train-mlogloss:0.540140	test-mlogloss:0.784826
# [154]	train-mlogloss:0.538657	test-mlogloss:0.784360
# [155]	train-mlogloss:0.537251	test-mlogloss:0.783959
# [156]	train-mlogloss:0.535802	test-mlogloss:0.783318
# [157]	train-mlogloss:0.534400	test-mlogloss:0.782895
# [158]	train-mlogloss:0.532918	test-mlogloss:0.782311
# [159]	train-mlogloss:0.531393	test-mlogloss:0.781815
# [160]	train-mlogloss:0.530060	test-mlogloss:0.781335
# [161]	train-mlogloss:0.528724	test-mlogloss:0.780901
# [162]	train-mlogloss:0.527493	test-mlogloss:0.780531
# [163]	train-mlogloss:0.526163	test-mlogloss:0.780000
# [164]	train-mlogloss:0.524929	test-mlogloss:0.779622
# [165]	train-mlogloss:0.523500	test-mlogloss:0.779101
# [166]	train-mlogloss:0.522159	test-mlogloss:0.778765
# [167]	train-mlogloss:0.520858	test-mlogloss:0.778446
# [168]	train-mlogloss:0.519579	test-mlogloss:0.778109
# [169]	train-mlogloss:0.518320	test-mlogloss:0.777724
# [170]	train-mlogloss:0.517008	test-mlogloss:0.777286
# [171]	train-mlogloss:0.515698	test-mlogloss:0.776770
# [172]	train-mlogloss:0.514452	test-mlogloss:0.776319
# [173]	train-mlogloss:0.513290	test-mlogloss:0.775907
# [174]	train-mlogloss:0.512120	test-mlogloss:0.775586
# [175]	train-mlogloss:0.510865	test-mlogloss:0.775210
# [176]	train-mlogloss:0.509586	test-mlogloss:0.774865
# [177]	train-mlogloss:0.508290	test-mlogloss:0.774511
# [178]	train-mlogloss:0.507008	test-mlogloss:0.774136
# [179]	train-mlogloss:0.505881	test-mlogloss:0.773747
# [180]	train-mlogloss:0.504670	test-mlogloss:0.773389
# [181]	train-mlogloss:0.503531	test-mlogloss:0.773031
# [182]	train-mlogloss:0.502389	test-mlogloss:0.772602
# [183]	train-mlogloss:0.501181	test-mlogloss:0.772310
# [184]	train-mlogloss:0.500015	test-mlogloss:0.772024
# [185]	train-mlogloss:0.498803	test-mlogloss:0.771761
# [186]	train-mlogloss:0.497676	test-mlogloss:0.771363
# [187]	train-mlogloss:0.496496	test-mlogloss:0.771008
# [188]	train-mlogloss:0.495255	test-mlogloss:0.770684
# [189]	train-mlogloss:0.494128	test-mlogloss:0.770398
# [190]	train-mlogloss:0.492994	test-mlogloss:0.770088
# [191]	train-mlogloss:0.491800	test-mlogloss:0.769729
# [192]	train-mlogloss:0.490608	test-mlogloss:0.769384
# [193]	train-mlogloss:0.489490	test-mlogloss:0.769027
# [194]	train-mlogloss:0.488345	test-mlogloss:0.768684
# [195]	train-mlogloss:0.487274	test-mlogloss:0.768326
# [196]	train-mlogloss:0.486223	test-mlogloss:0.768062
# [197]	train-mlogloss:0.485070	test-mlogloss:0.767727
# [198]	train-mlogloss:0.483873	test-mlogloss:0.767368
# [199]	train-mlogloss:0.482785	test-mlogloss:0.766991
# [200]	train-mlogloss:0.481620	test-mlogloss:0.766744
# [201]	train-mlogloss:0.480545	test-mlogloss:0.766488
# [202]	train-mlogloss:0.479435	test-mlogloss:0.766149
# [203]	train-mlogloss:0.478327	test-mlogloss:0.765885
# [204]	train-mlogloss:0.477310	test-mlogloss:0.765587
# [205]	train-mlogloss:0.476252	test-mlogloss:0.765284
# [206]	train-mlogloss:0.475293	test-mlogloss:0.765041
# [207]	train-mlogloss:0.474247	test-mlogloss:0.764740
# [208]	train-mlogloss:0.473170	test-mlogloss:0.764420
# [209]	train-mlogloss:0.472185	test-mlogloss:0.764181
# [210]	train-mlogloss:0.471100	test-mlogloss:0.763892
# [211]	train-mlogloss:0.469965	test-mlogloss:0.763612
# [212]	train-mlogloss:0.468929	test-mlogloss:0.763359
# [213]	train-mlogloss:0.467780	test-mlogloss:0.763021
# [214]	train-mlogloss:0.466751	test-mlogloss:0.762727
# [215]	train-mlogloss:0.465709	test-mlogloss:0.762520
# [216]	train-mlogloss:0.464826	test-mlogloss:0.762312
# [217]	train-mlogloss:0.463956	test-mlogloss:0.762028
# [218]	train-mlogloss:0.462860	test-mlogloss:0.761676
# [219]	train-mlogloss:0.461794	test-mlogloss:0.761308
# [220]	train-mlogloss:0.460788	test-mlogloss:0.761131
# [221]	train-mlogloss:0.459810	test-mlogloss:0.760920
# [222]	train-mlogloss:0.458901	test-mlogloss:0.760718
# [223]	train-mlogloss:0.457993	test-mlogloss:0.760486
# [224]	train-mlogloss:0.457024	test-mlogloss:0.760224
# [225]	train-mlogloss:0.456001	test-mlogloss:0.759933
# [226]	train-mlogloss:0.455133	test-mlogloss:0.759742
# [227]	train-mlogloss:0.454177	test-mlogloss:0.759485
# [228]	train-mlogloss:0.453221	test-mlogloss:0.759252
# [229]	train-mlogloss:0.452390	test-mlogloss:0.759045
# [230]	train-mlogloss:0.451422	test-mlogloss:0.758932
# [231]	train-mlogloss:0.450506	test-mlogloss:0.758688
# [232]	train-mlogloss:0.449454	test-mlogloss:0.758493
# [233]	train-mlogloss:0.448418	test-mlogloss:0.758264
# [234]	train-mlogloss:0.447505	test-mlogloss:0.758055
# [235]	train-mlogloss:0.446570	test-mlogloss:0.757807
# [236]	train-mlogloss:0.445668	test-mlogloss:0.757595
# [237]	train-mlogloss:0.444702	test-mlogloss:0.757363
# [238]	train-mlogloss:0.443861	test-mlogloss:0.757135
# [239]	train-mlogloss:0.442951	test-mlogloss:0.756982
# [240]	train-mlogloss:0.442107	test-mlogloss:0.756784
# [241]	train-mlogloss:0.441137	test-mlogloss:0.756524
# [242]	train-mlogloss:0.440184	test-mlogloss:0.756284
# [243]	train-mlogloss:0.439336	test-mlogloss:0.756067
# [244]	train-mlogloss:0.438443	test-mlogloss:0.755882
# [245]	train-mlogloss:0.437635	test-mlogloss:0.755713
# [246]	train-mlogloss:0.436742	test-mlogloss:0.755500
# [247]	train-mlogloss:0.435886	test-mlogloss:0.755275
# [248]	train-mlogloss:0.435015	test-mlogloss:0.755111
# [249]	train-mlogloss:0.434055	test-mlogloss:0.754956
# [250]	train-mlogloss:0.433153	test-mlogloss:0.754715
# [251]	train-mlogloss:0.432274	test-mlogloss:0.754550
# [252]	train-mlogloss:0.431204	test-mlogloss:0.754327
# [253]	train-mlogloss:0.430218	test-mlogloss:0.754054
# [254]	train-mlogloss:0.429395	test-mlogloss:0.753900
# [255]	train-mlogloss:0.428551	test-mlogloss:0.753660
# [256]	train-mlogloss:0.427724	test-mlogloss:0.753471
# [257]	train-mlogloss:0.426808	test-mlogloss:0.753183
# [258]	train-mlogloss:0.425888	test-mlogloss:0.753003
# [259]	train-mlogloss:0.425041	test-mlogloss:0.752755
# [260]	train-mlogloss:0.424195	test-mlogloss:0.752558
# [261]	train-mlogloss:0.423272	test-mlogloss:0.752298
# [262]	train-mlogloss:0.422416	test-mlogloss:0.752047
# [263]	train-mlogloss:0.421618	test-mlogloss:0.751904
# [264]	train-mlogloss:0.420824	test-mlogloss:0.751781
# [265]	train-mlogloss:0.420029	test-mlogloss:0.751631
# [266]	train-mlogloss:0.419211	test-mlogloss:0.751489
# [267]	train-mlogloss:0.418389	test-mlogloss:0.751425
# [268]	train-mlogloss:0.417632	test-mlogloss:0.751225
# [269]	train-mlogloss:0.416894	test-mlogloss:0.751107
# [270]	train-mlogloss:0.416129	test-mlogloss:0.750982
# [271]	train-mlogloss:0.415315	test-mlogloss:0.750812
# [272]	train-mlogloss:0.414552	test-mlogloss:0.750641
# [273]	train-mlogloss:0.413760	test-mlogloss:0.750531
# [274]	train-mlogloss:0.413038	test-mlogloss:0.750417
# [275]	train-mlogloss:0.412254	test-mlogloss:0.750185
# [276]	train-mlogloss:0.411511	test-mlogloss:0.750004
# [277]	train-mlogloss:0.410776	test-mlogloss:0.749837
# [278]	train-mlogloss:0.410067	test-mlogloss:0.749692
# [279]	train-mlogloss:0.409269	test-mlogloss:0.749535
# [280]	train-mlogloss:0.408535	test-mlogloss:0.749354
# [281]	train-mlogloss:0.407761	test-mlogloss:0.749196
# [282]	train-mlogloss:0.406986	test-mlogloss:0.749012
# [283]	train-mlogloss:0.406269	test-mlogloss:0.748908
# [284]	train-mlogloss:0.405492	test-mlogloss:0.748827
# [285]	train-mlogloss:0.404776	test-mlogloss:0.748686
# [286]	train-mlogloss:0.404086	test-mlogloss:0.748544
# [287]	train-mlogloss:0.403458	test-mlogloss:0.748459
# [288]	train-mlogloss:0.402822	test-mlogloss:0.748296
# [289]	train-mlogloss:0.402196	test-mlogloss:0.748181
# [290]	train-mlogloss:0.401484	test-mlogloss:0.748007
# [291]	train-mlogloss:0.400802	test-mlogloss:0.747854
# [292]	train-mlogloss:0.400067	test-mlogloss:0.747770
# [293]	train-mlogloss:0.399303	test-mlogloss:0.747676
# [294]	train-mlogloss:0.398637	test-mlogloss:0.747542
# [295]	train-mlogloss:0.397997	test-mlogloss:0.747421
# [296]	train-mlogloss:0.397285	test-mlogloss:0.747277
# [297]	train-mlogloss:0.396546	test-mlogloss:0.747145
# [298]	train-mlogloss:0.395920	test-mlogloss:0.747066
# [299]	train-mlogloss:0.395178	test-mlogloss:0.746922
# [300]	train-mlogloss:0.394449	test-mlogloss:0.746849
# [301]	train-mlogloss:0.393749	test-mlogloss:0.746708
# [302]	train-mlogloss:0.393019	test-mlogloss:0.746586
# [303]	train-mlogloss:0.392381	test-mlogloss:0.746508
# [304]	train-mlogloss:0.391621	test-mlogloss:0.746344
# [305]	train-mlogloss:0.390954	test-mlogloss:0.746228
# [306]	train-mlogloss:0.390284	test-mlogloss:0.746143
# [307]	train-mlogloss:0.389558	test-mlogloss:0.745950
# [308]	train-mlogloss:0.388844	test-mlogloss:0.745877
# [309]	train-mlogloss:0.388203	test-mlogloss:0.745766
# [310]	train-mlogloss:0.387533	test-mlogloss:0.745615
# [311]	train-mlogloss:0.386806	test-mlogloss:0.745517
# [312]	train-mlogloss:0.386178	test-mlogloss:0.745456
# [313]	train-mlogloss:0.385524	test-mlogloss:0.745386
# [314]	train-mlogloss:0.384851	test-mlogloss:0.745297
# [315]	train-mlogloss:0.384218	test-mlogloss:0.745163
# [316]	train-mlogloss:0.383523	test-mlogloss:0.744993
# [317]	train-mlogloss:0.382874	test-mlogloss:0.744959
# [318]	train-mlogloss:0.382275	test-mlogloss:0.744811
# [319]	train-mlogloss:0.381564	test-mlogloss:0.744754
# [320]	train-mlogloss:0.380907	test-mlogloss:0.744673
# [321]	train-mlogloss:0.380281	test-mlogloss:0.744484
# [322]	train-mlogloss:0.379641	test-mlogloss:0.744301
# [323]	train-mlogloss:0.378976	test-mlogloss:0.744183
# [324]	train-mlogloss:0.378346	test-mlogloss:0.744022
# [325]	train-mlogloss:0.377748	test-mlogloss:0.743909
# [326]	train-mlogloss:0.377127	test-mlogloss:0.743857
# [327]	train-mlogloss:0.376466	test-mlogloss:0.743790
# [328]	train-mlogloss:0.375860	test-mlogloss:0.743690
# [329]	train-mlogloss:0.375221	test-mlogloss:0.743641
# [330]	train-mlogloss:0.374538	test-mlogloss:0.743534
# [331]	train-mlogloss:0.373825	test-mlogloss:0.743446
# [332]	train-mlogloss:0.373282	test-mlogloss:0.743290
# [333]	train-mlogloss:0.372605	test-mlogloss:0.743189
# [334]	train-mlogloss:0.371987	test-mlogloss:0.743078
# [335]	train-mlogloss:0.371471	test-mlogloss:0.742994
# [336]	train-mlogloss:0.370813	test-mlogloss:0.742944
# [337]	train-mlogloss:0.370211	test-mlogloss:0.742795
# [338]	train-mlogloss:0.369688	test-mlogloss:0.742715
# [339]	train-mlogloss:0.369091	test-mlogloss:0.742551
# [340]	train-mlogloss:0.368599	test-mlogloss:0.742531
# [341]	train-mlogloss:0.367931	test-mlogloss:0.742457
# [342]	train-mlogloss:0.367299	test-mlogloss:0.742302
# [343]	train-mlogloss:0.366723	test-mlogloss:0.742238
# [344]	train-mlogloss:0.366075	test-mlogloss:0.742107
# [345]	train-mlogloss:0.365521	test-mlogloss:0.742089
# [346]	train-mlogloss:0.364951	test-mlogloss:0.741974
# [347]	train-mlogloss:0.364389	test-mlogloss:0.741929
# [348]	train-mlogloss:0.363845	test-mlogloss:0.741879
# [349]	train-mlogloss:0.363239	test-mlogloss:0.741735
# [350]	train-mlogloss:0.362678	test-mlogloss:0.741689
# [351]	train-mlogloss:0.361992	test-mlogloss:0.741551
# [352]	train-mlogloss:0.361431	test-mlogloss:0.741516
# [353]	train-mlogloss:0.360854	test-mlogloss:0.741490
# [354]	train-mlogloss:0.360331	test-mlogloss:0.741393
# [355]	train-mlogloss:0.359686	test-mlogloss:0.741261
# [356]	train-mlogloss:0.359077	test-mlogloss:0.741142
# [357]	train-mlogloss:0.358506	test-mlogloss:0.741093
# [358]	train-mlogloss:0.357932	test-mlogloss:0.741004
# [359]	train-mlogloss:0.357322	test-mlogloss:0.740917
# [360]	train-mlogloss:0.356761	test-mlogloss:0.740844
# [361]	train-mlogloss:0.356064	test-mlogloss:0.740737
# [362]	train-mlogloss:0.355488	test-mlogloss:0.740658
# [363]	train-mlogloss:0.354969	test-mlogloss:0.740551
# [364]	train-mlogloss:0.354513	test-mlogloss:0.740469
# [365]	train-mlogloss:0.353936	test-mlogloss:0.740389
# [366]	train-mlogloss:0.353399	test-mlogloss:0.740293
# [367]	train-mlogloss:0.352811	test-mlogloss:0.740194
# [368]	train-mlogloss:0.352309	test-mlogloss:0.740080
# [369]	train-mlogloss:0.351770	test-mlogloss:0.740019
# [370]	train-mlogloss:0.351208	test-mlogloss:0.739969
# [371]	train-mlogloss:0.350653	test-mlogloss:0.739930
# [372]	train-mlogloss:0.350115	test-mlogloss:0.739949
# [373]	train-mlogloss:0.349587	test-mlogloss:0.739919
# [374]	train-mlogloss:0.349019	test-mlogloss:0.739889
# [375]	train-mlogloss:0.348457	test-mlogloss:0.739823
# [376]	train-mlogloss:0.347922	test-mlogloss:0.739797
# [377]	train-mlogloss:0.347338	test-mlogloss:0.739741
# [378]	train-mlogloss:0.346775	test-mlogloss:0.739671
# [379]	train-mlogloss:0.346102	test-mlogloss:0.739566
# [380]	train-mlogloss:0.345620	test-mlogloss:0.739489
# [381]	train-mlogloss:0.345041	test-mlogloss:0.739447
# [382]	train-mlogloss:0.344467	test-mlogloss:0.739408
# [383]	train-mlogloss:0.343929	test-mlogloss:0.739359
# [384]	train-mlogloss:0.343437	test-mlogloss:0.739349
# [385]	train-mlogloss:0.342977	test-mlogloss:0.739316
# [386]	train-mlogloss:0.342467	test-mlogloss:0.739316
# [387]	train-mlogloss:0.341998	test-mlogloss:0.739276
# [388]	train-mlogloss:0.341471	test-mlogloss:0.739285
# [389]	train-mlogloss:0.340899	test-mlogloss:0.739207
# [390]	train-mlogloss:0.340325	test-mlogloss:0.739077
# [391]	train-mlogloss:0.339800	test-mlogloss:0.739084
# [392]	train-mlogloss:0.339295	test-mlogloss:0.738948
# [393]	train-mlogloss:0.338769	test-mlogloss:0.738874
# [394]	train-mlogloss:0.338227	test-mlogloss:0.738818
# [395]	train-mlogloss:0.337730	test-mlogloss:0.738799
# [396]	train-mlogloss:0.337250	test-mlogloss:0.738779
# [397]	train-mlogloss:0.336695	test-mlogloss:0.738702
# [398]	train-mlogloss:0.336136	test-mlogloss:0.738645
# [399]	train-mlogloss:0.335613	test-mlogloss:0.738622
# [400]	train-mlogloss:0.335122	test-mlogloss:0.738589
# [401]	train-mlogloss:0.334641	test-mlogloss:0.738532
# [402]	train-mlogloss:0.334110	test-mlogloss:0.738441
# [403]	train-mlogloss:0.333582	test-mlogloss:0.738368
# [404]	train-mlogloss:0.333074	test-mlogloss:0.738322
# [405]	train-mlogloss:0.332555	test-mlogloss:0.738246
# [406]	train-mlogloss:0.332069	test-mlogloss:0.738243
# [407]	train-mlogloss:0.331598	test-mlogloss:0.738236
# [408]	train-mlogloss:0.331192	test-mlogloss:0.738239
# [409]	train-mlogloss:0.330729	test-mlogloss:0.738182
# [410]	train-mlogloss:0.330247	test-mlogloss:0.738137
# [411]	train-mlogloss:0.329809	test-mlogloss:0.738069
# [412]	train-mlogloss:0.329345	test-mlogloss:0.738072
# [413]	train-mlogloss:0.328861	test-mlogloss:0.738084
# [414]	train-mlogloss:0.328402	test-mlogloss:0.738034
# [415]	train-mlogloss:0.327901	test-mlogloss:0.737982
# [416]	train-mlogloss:0.327390	test-mlogloss:0.737934
# [417]	train-mlogloss:0.326847	test-mlogloss:0.737893
# [418]	train-mlogloss:0.326359	test-mlogloss:0.737891
# [419]	train-mlogloss:0.325864	test-mlogloss:0.737830
# [420]	train-mlogloss:0.325413	test-mlogloss:0.737807
# [421]	train-mlogloss:0.324912	test-mlogloss:0.737727
# [422]	train-mlogloss:0.324447	test-mlogloss:0.737679
# [423]	train-mlogloss:0.324050	test-mlogloss:0.737670
# [424]	train-mlogloss:0.323553	test-mlogloss:0.737641
# [425]	train-mlogloss:0.323084	test-mlogloss:0.737577
# [426]	train-mlogloss:0.322611	test-mlogloss:0.737577
# [427]	train-mlogloss:0.322144	test-mlogloss:0.737538
# [428]	train-mlogloss:0.321687	test-mlogloss:0.737526
# [429]	train-mlogloss:0.321218	test-mlogloss:0.737483
# [430]	train-mlogloss:0.320790	test-mlogloss:0.737491
# [431]	train-mlogloss:0.320338	test-mlogloss:0.737457
# [432]	train-mlogloss:0.319856	test-mlogloss:0.737418
# [433]	train-mlogloss:0.319430	test-mlogloss:0.737434
# [434]	train-mlogloss:0.318999	test-mlogloss:0.737449
# [435]	train-mlogloss:0.318463	test-mlogloss:0.737434
# [436]	train-mlogloss:0.318006	test-mlogloss:0.737415
# [437]	train-mlogloss:0.317557	test-mlogloss:0.737366
# [438]	train-mlogloss:0.317160	test-mlogloss:0.737335
# [439]	train-mlogloss:0.316748	test-mlogloss:0.737343
# [440]	train-mlogloss:0.316275	test-mlogloss:0.737317
# [441]	train-mlogloss:0.315769	test-mlogloss:0.737251
# [442]	train-mlogloss:0.315336	test-mlogloss:0.737227
# [443]	train-mlogloss:0.314856	test-mlogloss:0.737185
# [444]	train-mlogloss:0.314426	test-mlogloss:0.737146
# [445]	train-mlogloss:0.313943	test-mlogloss:0.737190
# [446]	train-mlogloss:0.313494	test-mlogloss:0.737158
# [447]	train-mlogloss:0.313000	test-mlogloss:0.737205
# [448]	train-mlogloss:0.312540	test-mlogloss:0.737157
# [449]	train-mlogloss:0.312062	test-mlogloss:0.737134
# [450]	train-mlogloss:0.311531	test-mlogloss:0.737105
# [451]	train-mlogloss:0.311112	test-mlogloss:0.737034
# [452]	train-mlogloss:0.310681	test-mlogloss:0.737021
# [453]	train-mlogloss:0.310175	test-mlogloss:0.736949
# [454]	train-mlogloss:0.309743	test-mlogloss:0.736917
# [455]	train-mlogloss:0.309315	test-mlogloss:0.736846
# [456]	train-mlogloss:0.308863	test-mlogloss:0.736859
# [457]	train-mlogloss:0.308421	test-mlogloss:0.736860
# [458]	train-mlogloss:0.308006	test-mlogloss:0.736797
# [459]	train-mlogloss:0.307523	test-mlogloss:0.736774
# [460]	train-mlogloss:0.307091	test-mlogloss:0.736764
# [461]	train-mlogloss:0.306655	test-mlogloss:0.736694
# [462]	train-mlogloss:0.306157	test-mlogloss:0.736736
# [463]	train-mlogloss:0.305708	test-mlogloss:0.736760
# [464]	train-mlogloss:0.305270	test-mlogloss:0.736710
# [465]	train-mlogloss:0.304842	test-mlogloss:0.736714
# [466]	train-mlogloss:0.304426	test-mlogloss:0.736697
# [467]	train-mlogloss:0.303945	test-mlogloss:0.736680
# [468]	train-mlogloss:0.303521	test-mlogloss:0.736647
# [469]	train-mlogloss:0.303089	test-mlogloss:0.736652
# [470]	train-mlogloss:0.302713	test-mlogloss:0.736681
# [471]	train-mlogloss:0.302253	test-mlogloss:0.736698
# [472]	train-mlogloss:0.301906	test-mlogloss:0.736717
# [473]	train-mlogloss:0.301500	test-mlogloss:0.736741
# [474]	train-mlogloss:0.301065	test-mlogloss:0.736696
# [475]	train-mlogloss:0.300660	test-mlogloss:0.736710
# [476]	train-mlogloss:0.300262	test-mlogloss:0.736697
# [477]	train-mlogloss:0.299806	test-mlogloss:0.736769
# [478]	train-mlogloss:0.299348	test-mlogloss:0.736733
# [479]	train-mlogloss:0.298994	test-mlogloss:0.736752
# [480]	train-mlogloss:0.298646	test-mlogloss:0.736776
# [481]	train-mlogloss:0.298193	test-mlogloss:0.736728
# [482]	train-mlogloss:0.297796	test-mlogloss:0.736758
# [483]	train-mlogloss:0.297406	test-mlogloss:0.736732
# [484]	train-mlogloss:0.296967	test-mlogloss:0.736697
# [485]	train-mlogloss:0.296573	test-mlogloss:0.736736
# [486]	train-mlogloss:0.296106	test-mlogloss:0.736652
# [487]	train-mlogloss:0.295664	test-mlogloss:0.736676
# [488]	train-mlogloss:0.295170	test-mlogloss:0.736647
# [489]	train-mlogloss:0.294776	test-mlogloss:0.736640
# [490]	train-mlogloss:0.294396	test-mlogloss:0.736674
# [491]	train-mlogloss:0.294028	test-mlogloss:0.736724
# [492]	train-mlogloss:0.293623	test-mlogloss:0.736716
# [493]	train-mlogloss:0.293219	test-mlogloss:0.736698
# [494]	train-mlogloss:0.292774	test-mlogloss:0.736679
# [495]	train-mlogloss:0.292414	test-mlogloss:0.736673
# [496]	train-mlogloss:0.292052	test-mlogloss:0.736602
# [497]	train-mlogloss:0.291653	test-mlogloss:0.736553
# [498]	train-mlogloss:0.291313	test-mlogloss:0.736556
# [499]	train-mlogloss:0.290916	test-mlogloss:0.736530
# The log loss is:  0.736529959845
# The best metric is: 0.736529959845 for the params: {'num_class': 38, 'silent': 1, 'eval_metric': 'mlogloss', 'nthread': 4, 'objective': 'multi:softprob', 'eta': 0.1, 'num_round': 500, 'max_depth': 4, 'chi2_lim': 250}
#
# Process finished with exit code 0
