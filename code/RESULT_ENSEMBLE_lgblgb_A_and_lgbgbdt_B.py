#!/usr/bin/python
# -*- coding: utf-8 -*-
# date: 2017
# author: Kyz
# desc: New

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    line = range(13463)

    data5885 = pd.read_csv(u'../result/GBDT线上0.5885用了LGB特征选择/B_test_fslgb400_predict_without_cv_fillna_10_rd_0_GBDT_N_400_features_109.csv'
                           , index_col='no')
    # data584814 = pd.read_csv(u'../result/GBDT建模线上0.584814用了XGB特征选择/B_test_predict_fsxgbN300_without_cv_fillna_10_rd_0_GBDT_N_400_features_108.csv',
    #                          index_col = 'no')
    # data584806 = pd.read_csv(u'../result/GBDT线上0.584806的结果GBDT特征选择/B_test_final_predict_fillna_10_rd_0_GBDT_N_400_features_142.csv',
    #                          index_col = 'no')

    data_use_A_offline_577 = pd.read_csv('../result/B_test_predict_using_A_without_cv_fillna_10__N_400_features_256_offline_0.577045449978.csv',
                                         index_col='no')

    data_use_A_GBDT_offline_578 = pd.read_csv('../result/B_test_predict_using_A_without_cv_fillna_10__N_138_features_192_offline_0.578309517635.csv',
                                              index_col='no')

    # data5885.plot(x='no', y='pred', kind='scatter')
    # data584814.plot(x='no', y='pred', kind='scatter')
    # data584806.plot(x='no', y='pred', kind='scatter')

    # plt.figure(1)
    # plt.scatter(x=line, y=data5885['pred'], color='blue')
    # plt.scatter(x=line, y=data584814['pred'])
    # plt.scatter(x=line, y=data584806['pred'])


    # data5885['pred'].plot(kind='bar')
    # data584814['pred']
    # data584806['pred']

    # plt.show()

    # data_save = data584806
    #
    # avg = data_use_A_offline_577['pred'] * 0.16 + data5885['pred'] * 0.84
    # data_save['pred'] = avg
    # # data_save.to_csv('../result/weighted_avg_of_16_mA_offline_577_and_84_m5885.csv')
    # print avg
    #
    # plt.figure(2)
    # plt.scatter(x=line, y=avg, edgecolors='green')

    # data_save1 = data584806

    avg1 = data_use_A_offline_577['pred'] * 0.11 + data5885['pred'] * 0.89
    data_save1['pred'] = avg1
    data_save1.to_csv('../result/weighted_avg_of_11_mA_offline_577_and_89_m5885.csv')
    temp = []
    for i in range(13463):
        if avg1.values[i] >= 0.085:
            temp.append(avg1.values[i])
    print len(temp)
    # print avg1

    # plt.figure(3)
    # plt.scatter(x=line, y=avg1)
    # plt.show()
