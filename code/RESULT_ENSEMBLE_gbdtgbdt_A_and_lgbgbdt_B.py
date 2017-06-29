#!/usr/bin/python
# -*- coding: utf-8 -*-
# date: 2017
# author: Kyz
# desc: 融合B_lgbgbdt和A_gbdtgbdt，线上0.606803

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    line = range(13463)

    data5885 = pd.read_csv(u'../result/GBDT线上0.5885用了LGB特征选择/B_test_fslgb400_predict_without_cv_fillna_10_rd_0_GBDT_N_400_features_109.csv'
                           , index_col='no')
    data584806 = pd.read_csv(u'../result/GBDT线上0.584806的结果GBDT特征选择/B_test_final_predict_fillna_10_rd_0_GBDT_N_400_features_142.csv',
                             index_col='no')

    data_use_A_GBDT_offline_5805 = pd.read_csv('../result/B_test_2fs_using_A_GBDT_without_cv_fillna_1_N_141_features_192_offline_0.580506564827.csv',
                                               index_col='no')

    data_save1 = data584806

    w1 = 0.245
    w2 = 0.755

    avg1 = data_use_A_GBDT_offline_5805['pred'] * w1 + data5885['pred'] * w2
    data_save1['pred'] = avg1
    temp = []
    for i in range(13463):
        if avg1.values[i] >= 0.085:
            temp.append(avg1.values[i])
    print len(temp)
    data_save1.to_csv('../result/weighted_avg_of_' + str(w1) + '_Agbdtbe_' + str(w2) + '_m5885_' + str(len(temp))+'.csv')
    # print avg1

    # plt.figure(3)
    # plt.scatter(x=line, y=avg1)
    # # plt.show()
