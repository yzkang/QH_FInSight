#!/usr/bin/python
# -*- coding: utf-8 -*-
# date: 2017
# author: Kyz
# desc: 这个是产生线上AUC最好结果0.6075的融合代码，比例是0.2和0.8

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    line = range(13463)

    data5996 = pd.read_csv('../result/Final/weighted_avg_of_10_mA_lgb_and_90_m5885_online_0.5996.csv'
                           , index_col='no')
    data6803 = pd.read_csv('../result/weighted_avg_of_0.24_Agbdtbe_0.76_m5885_5975.csv',
                             index_col = 'no')

    data_save1 = data6803

    w1 = 0.2
    w2 = 0.8

    avg1 = data5996['pred'] * w1 + data6803['pred'] * w2
    data_save1['pred'] = avg1
    temp = []
    for i in range(13463):
        if avg1.values[i] >= 0.085:
            temp.append(avg1.values[i])
            # print avg1.values[i]
    print len(temp)
    data_save1.to_csv('../result/weighted_avg_of_'+str(w1)+'_Algb_'+str(w2)+'_Agbdtbestnew606832_'+str(len(temp))+'.csv')

    # plt.figure(1)
    # plt.scatter(x=line, y=avg1, edgecolors='red')

