#!/usr/bin/python
#  -*- coding: utf-8 -*-
# date: 2017
# author: Kang Yan Zhe

import csv
import time
import pandas as pd
import numpy as np
from scipy import interp
from math import isnan
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def lgb_feature_selection(fe_name, matrix_x_temp, label_y, th):
    # SelectfromModel
    clf = LGBMClassifier(n_estimators=400)
    clf.fit(matrix_x_temp, label_y)
    sfm = SelectFromModel(clf, prefit=True, threshold=th)
    matrix_x = sfm.transform(matrix_x_temp)

    # 打印出有多少特征重要性非零的特征
    feature_score_dict = {}
    for fn, s in zip(fe_name, clf.feature_importances_):
        feature_score_dict[fn] = s
    m = 0
    for k in feature_score_dict:
        if feature_score_dict[k] == 0.0:
            m += 1
    print 'number of not-zero features:' + str(len(feature_score_dict) - m)

    # 打印出特征重要性
    feature_score_dict_sorted = sorted(feature_score_dict.items(),
                                       key=lambda d: d[1], reverse=True)
    print 'feature_importance:'
    for ii in range(len(feature_score_dict_sorted)):
        print feature_score_dict_sorted[ii][0], feature_score_dict_sorted[ii][1]
    print '\n'

    f = open('../eda/lgb_feature_importance.txt', 'w')
    f.write(th)
    f.write('\nRank\tFeature Name\tFeature Importance\n')
    for i in range(len(feature_score_dict_sorted)):
        f.write(str(i) + '\t' + str(feature_score_dict_sorted[i][0]) + '\t' + str(feature_score_dict_sorted[i][1]) + '\n')
    f.close()

    # 打印具体使用了哪些字段
    how_long = matrix_x.shape[1]  # matrix_x 是 特征选择后的 输入矩阵
    feature_used_dict_temp = feature_score_dict_sorted[:how_long]
    feature_used_name = []
    for ii in range(len(feature_used_dict_temp)):
        feature_used_name.append(feature_used_dict_temp[ii][0])
    print 'feature_chooesed:'
    for ii in range(len(feature_used_name)):
        print feature_used_name[ii]
    print '\n'

    f = open('../eda/lgb_feature_chose.txt', 'w')
    f.write('Feature Chose Name :\n')
    for i in range(len(feature_used_name)):
        f.write(str(feature_used_name[i]) + '\n')
    f.close()

    # 找到未被使用的字段名
    feature_not_used_name = []
    for i in range(len(fe_name)):
        if fe_name[i] not in feature_used_name:
            feature_not_used_name.append(fe_name[i])

    return matrix_x, feature_not_used_name[:], len(feature_used_name)


def data_test_feature_drop(data_test, feature_name_drop):
    for col in feature_name_drop:
        data_test.drop(col, axis=1, inplace=True)
    print data_test.shape
    return data_test.as_matrix()


def write_predict_results_to_csv(csv_name, uid, prob_list):

    csv_file = file(csv_name, 'wb')
    writer = csv.writer(csv_file)
    combined_list = [['no', 'pred']]
    if len(uid) == len(prob_list):
        for i in range(len(uid)):
            combined_list.append([str(uid[i]), str(prob_list[i])])
        writer.writerows(combined_list)
        csv_file.close()
    else:
        print 'no和pred的个数不一致'


def without_cv_transfer_a_to_b_modeling():
    """

    :return:
    """

    '''Data input'''
    data_a_train = pd.read_csv('../data/A_train_final.csv', index_col='no')
    data_b_train = pd.read_csv('../data/B_train_final.csv', index_col='no')
    y_of_b_train = data_b_train['flag']
    data_b_test = pd.read_csv('../data/B_test_final.csv', index_col='no')

    '''A train特征工程'''
    data_a_train_without_label = data_a_train.drop('flag', axis=1)

    data_a_train_without_label['UserInfo_222x82'] = data_a_train_without_label['UserInfo_82'] * data_a_train_without_label['UserInfo_222']

    '''缺失值填充'''
    data_a_train_filled = data_a_train_without_label.fillna(value=10)

    '''特征的名字'''
    feature_name = list(data_a_train_without_label.columns.values)
    data_b_test_user_id = list(data_b_test.index.values)

    '''构造训练集和测试集'''
    x_temp = data_a_train_filled.iloc[:, :].as_matrix()  # 自变量
    y = data_a_train.iloc[:, -1].as_matrix()  # 因变量

    '''Feature selection 注意如果加特征的话，feature name还是需要改的'''
    X, dropped_feature_name, len_feature_choose = lgb_feature_selection(feature_name, x_temp, y, "0.1*mean")

    '''B train特征工程'''
    data_b_train_without_label = data_b_train.drop('flag', axis=1)

    data_b_train_without_label['UserInfo_222x82'] = data_b_train_without_label['UserInfo_82'] * data_b_train_without_label['UserInfo_222']
    data_b_train_filled = data_b_train_without_label.fillna(value=10)

    '''b test 特征工程'''
    data_b_test['UserInfo_222x82'] = data_b_test['UserInfo_82'] * data_b_test['UserInfo_222']
    data_b_test_filled = data_b_test.fillna(value=10)

    '''特征筛选'''
    data_b_train_filled_after_feature_selection = data_test_feature_drop(data_b_train_filled, dropped_feature_name)
    data_b_test_filled_after_feature_selection = data_test_feature_drop(data_b_test_filled, dropped_feature_name)

    '''用A_train建模预测B_train'''

    print '起始时间'
    print time.clock()*1.0/60

    parameter_n_estimators = 400
    classifier = LGBMClassifier(n_estimators=parameter_n_estimators)

    a_model = classifier.fit(X, y)

    prob_of_b_train = a_model.predict_proba(data_b_train_filled_after_feature_selection)

    print '训练终止时间'
    print time.clock()*1.0/60

    '''画roc曲线'''
    fpr, tpr, thresholds = roc_curve(y_of_b_train, prob_of_b_train[:, 1])

    roc_auc = auc(fpr, tpr)

    print '\nauc='+str(roc_auc)

    '''预测Btest'''

    prob_of_b_test = a_model.predict_proba(data_b_test_filled_after_feature_selection)

    result_file_name = '../result/B_test_predict_using_A_LGBLGB_without_cv_fillna_10' + '_N_' + str(parameter_n_estimators) + '_features_' + \
                       str(len_feature_choose) + '_offline_'+str(roc_auc)+'.csv'

    write_predict_results_to_csv(result_file_name, data_b_test_user_id, prob_of_b_test[:, 1].tolist())

