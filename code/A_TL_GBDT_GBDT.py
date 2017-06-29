#!/usr/bin/python
#  -*- coding: utf-8 -*-
# date: 2017
# author: Kang Yan Zhe
# param: classifier = GradientBoostingClassifier(n_estimators=140, learning_rate=0.5, max_depth=2,
#                                                random_state=0, max_features=10, min_weight_fraction_leaf=0.15)


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


def gbdt_feature_selection(fe_name, matrix_x_temp, label_y, th):
    # SelectfromModel
    clf = GradientBoostingClassifier(n_estimators=200, random_state=100)
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

    f = open('../eda/A_gbdt_feature_importance.txt', 'w')
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

    f = open('../eda/A_gbdt_feature_chose.txt', 'w')
    f.write('Feature Chose Name :\n')
    for i in range(len(feature_used_name)):
        f.write(str(feature_used_name[i]) + '\n')
    f.close()

    # 找到未被使用的字段名
    feature_not_used_name = []
    for i in range(len(fe_name)):
        if fe_name[i] not in feature_used_name:
            feature_not_used_name.append(fe_name[i])

    # 生成一个染色体（诸如01011100这样的）
    chromosome_temp = ''
    feature_name_ivar = fe_name[:-1]
    for ii in range(len(feature_name_ivar)):
        if feature_name_ivar[ii] in feature_used_name:
            chromosome_temp += '1'
        else:
            chromosome_temp += '0'
    print 'Chromosome:'
    print chromosome_temp
    joblib.dump(chromosome_temp, '../config/chromosome.pkl')
    print '\n'
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


def gbdt_cv_parameter_tuning():
    """

    :return:
    """

    '''Data input'''
    data_a_train = pd.read_csv('../data/A_train_final.csv', index_col='no')
    data_test = pd.read_csv('../data/B_train_final.csv', index_col='no')

    data_train = data_a_train

    data_train_without_label = data_train.drop('flag', axis=1)
    frames = [data_train_without_label, data_test]
    data_all = pd.concat(frames)

    '''给定一个随机数种子，打乱train'''
    s = 0
    np.random.seed(s)
    sampler = np.random.permutation(len(data_train.values))
    data_train_randomized = data_train.take(sampler)

    feature_name = list(data_train.columns.values)
    data_test_user_id = list(data_test.index.values)

    '''缺失值填充'''
    data_train_filled = data_train_randomized.fillna(value=10)

    '''构造训练集和测试集'''
    x_temp = data_train_filled.iloc[:, :-1].as_matrix()  # 自变量
    y = data_train_filled.iloc[:, -1].as_matrix()  # 因变量

    '''Feature selection'''
    X, dropped_feature_name, len_feature_choose = gbdt_feature_selection(feature_name, x_temp, y, '0.1*mean')

    '''调参'''
    gbdt_model = GradientBoostingClassifier()

    param_grid = {'learning_rate': [0.1, 0.5, 1],
                  'n_estimators': [100, 110, 120, 130, 140, 150],
                  'subsample': [0.8, 1.0],
                  'min_samples_split': [2, 3],
                  'min_samples_leaf': [1, 2],
                  'max_depth': [2, 3, 4],
                  'min_weight_fraction_leaf': [0, 0.05, 0.1, 0.15, 0.2]
                  # 'random_state': [None, 0, 10, 100],
                  # 'max_features': [None, 'auto', 'log2', 10]
                  #
                  }

    scores = ['roc_auc']
    # 分别按照precision和recall去寻找
    for score in scores:
        print "调参起始时间"
        print time.clock()
        print('\n')
        print("# Tuning hyper-parameters for %s" % score)
        print('\n')

        clf = GridSearchCV(gbdt_model, param_grid=param_grid, cv=5, n_jobs=-1,
                           pre_dispatch='2*n_jobs', scoring='%s' % score)
        clf.fit(X, y)

        print "Best parameters set found on development set:"
        print clf.best_params_
        print '\n'
        print clf.cv_results_
        print '\n'
        print "Detailed classification report:"
        print '\n'
        print "The model is trained on the full development set."
        print "The scores are computed on the full evaluation set."
        print '\n'

    print time.clock()


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
    data_a_train_without_label['UserInfo_82_median'] = data_a_train_without_label['UserInfo_82'].fillna(data_a_train_without_label[
                                                                                                            'UserInfo_82'].median())

    '''缺失值填充'''
    data_a_train_filled = data_a_train_without_label.fillna(value=1)

    '''特征的名字'''
    feature_name = list(data_a_train_without_label.columns.values)
    data_b_test_user_id = list(data_b_test.index.values)

    '''构造训练集和测试集'''
    x_temp = data_a_train_filled.iloc[:, :].as_matrix()  # 自变量
    y = data_a_train.iloc[:, -1].as_matrix()  # 因变量

    '''Feature selection 注意如果加特征的话，feature name还是需要改的'''
    X, dropped_feature_name, len_feature_choose = gbdt_feature_selection(feature_name, x_temp, y, "0.1*mean")

    '''B train特征工程'''
    data_b_train_without_label = data_b_train.drop('flag', axis=1)

    data_b_train_without_label['UserInfo_222x82'] = data_b_train_without_label['UserInfo_82'] * data_b_train_without_label['UserInfo_222']
    data_b_train_without_label['UserInfo_82_median'] = data_b_train_without_label['UserInfo_82'].fillna(data_b_test['UserInfo_82'].median())
    data_b_train_filled = data_b_train_without_label.fillna(value=1)

    '''b test 特征工程'''

    data_b_test['UserInfo_222x82'] = data_b_test['UserInfo_82'] * data_b_test['UserInfo_222']
    data_b_test['UserInfo_82_median'] = data_b_test['UserInfo_82'].fillna(data_b_test['UserInfo_82'].median())

    data_b_test_filled = data_b_test.fillna(value=1)

    '''特征筛选'''
    data_b_train_filled_after_feature_selection = data_test_feature_drop(data_b_train_filled, dropped_feature_name)
    data_b_test_filled_after_feature_selection = data_test_feature_drop(data_b_test_filled, dropped_feature_name)

    '''用A_train建模预测B_train'''

    print '起始时间'
    print time.clock()*1.0/60

    parameter_n_estimators = 140
    classifier = GradientBoostingClassifier(n_estimators=140, learning_rate=0.5, max_depth=2, random_state=0, max_features=10,
                                            min_weight_fraction_leaf=0.15)  #

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

    result_file_name = '../result/B_test_2fs_using_A_GBDT_without_cv_fillna_1' + '_N_' + str(parameter_n_estimators) + '_features_' + \
                       str(len_feature_choose) + '_offline_'+str(roc_auc)+'.csv'

    write_predict_results_to_csv(result_file_name, data_b_test_user_id, prob_of_b_test[:, 1].tolist())

