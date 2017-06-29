#!/usr/bin/python
#  -*- coding: utf-8 -*-
# date: 2017
# author: Kang Yan Zhe
# desc:

import csv
import time
import pandas as pd
import numpy as np
from scipy import interp
from math import isnan
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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
    f.write('Rank\tFeature Name\tFeature Importance\n')
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
    return matrix_x, feature_not_used_name[:-1], len(feature_used_name)


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


def gbdt_cv_modeling():
    """

    :return:
    """

    '''Data input'''
    data_b_train = pd.read_csv('../data/B_train_final.csv', index_col='no')
    data_test = pd.read_csv('../data/B_test_final.csv', index_col='no')

    data_train = data_b_train

    data_train_without_label = data_train.drop('flag', axis=1)
    frames = [data_train_without_label, data_test]

    '''给定一个随机数种子，打乱train'''
    s = 0
    np.random.seed(s)
    sampler = np.random.permutation(len(data_train.values))
    data_train_randomized = data_train.take(sampler)

    feature_name = list(data_train.columns.values)
    '''缺失值填充'''
    data_train_filled = data_train_randomized.fillna(value=10)

    '''构造训练集和测试集'''
    x_temp = data_train_filled.iloc[:, :-1].as_matrix()  # 自变量
    y = data_train_filled.iloc[:, -1].as_matrix()  # 因变量

    '''Feature selection'''
    X, dropped_feature_name, len_feature_choose = lgb_feature_selection(feature_name, x_temp, y, '0.1*mean')

    '''处理 验证集 B_test'''
    data_test_filled = data_test.fillna(value=10)
    data_test_filled_after_feature_selection = data_test_feature_drop(data_test_filled, dropped_feature_name)

    '''Split train/test data sets'''
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  # 分层抽样  cv的意思是cross-validation

    '''Choose a classification model'''
    parameter_n_estimators = 400
    classifier = GradientBoostingClassifier(n_estimators=parameter_n_estimators)

    '''Model fit, predict and ROC'''
    colors = cycle(['cyan', 'indigo', 'seagreen', 'orange', 'blue'])
    lw = 2
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 500)
    i_of_roc = 0
    a = 0

    probability_set_of_b_test = []

    for (train_indice, test_indice), color in zip(cv.split(X, y), colors):
        a_model = classifier.fit(X[train_indice], y[train_indice])

        probas_ = a_model.predict_proba(X[test_indice])

        prob_of_b_test = a_model.predict_proba(data_test_filled_after_feature_selection)  # 对B_test进行预测

        probability_set_of_b_test.append(prob_of_b_test[:, 1])

        fpr, tpr, thresholds = roc_curve(y[test_indice], probas_[:, 1])

        a += 1  # 序号加1

        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.4f)' % (i_of_roc, roc_auc))
        i_of_roc += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')

    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print 'mean_auc=' + str(mean_auc)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label='Mean ROC (area = %0.4f)' % mean_auc, lw=lw)

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('ROC_rd_' + str(s) + '_gbdt_' + str(len_feature_choose) + '_features')
    plt.legend(loc="lower right")
    plt.show()

    avg_prob = (probability_set_of_b_test[0] + probability_set_of_b_test[1] + probability_set_of_b_test[2] +
                probability_set_of_b_test[3] + probability_set_of_b_test[4]) * 1.0 / 5

    result_file_name = '../result/B_test_gbdt_predict_cv_fillna_10_rd_' + str(s) + '_N_' + str(parameter_n_estimators) + '_features_' + \
                       str(len_feature_choose) + '.csv'


def gbdt_cv_parameter_tuning():
    """

    :return:
    """

    '''Data input'''
    data_b_train = pd.read_csv('../data/B_train_final.csv', index_col='no')
    data_test = pd.read_csv('../data/B_test_final.csv', index_col='no')

    data_train = data_b_train

    data_train_without_label = data_train.drop('flag', axis=1)
    frames = [data_train_without_label, data_test]

    '''给定一个随机数种子，打乱train'''
    s = 0
    np.random.seed(s)  # 使用woe，未删缺失值，使用前300个字段，选择第0组模型的时候，得到了0.57的结果
    sampler = np.random.permutation(len(data_train.values))
    data_train_randomized = data_train.take(sampler)

    feature_name = list(data_train.columns.values)

    '''缺失值填充'''
    data_train_filled = data_train_randomized.fillna(value=10)

    '''构造训练集和测试集'''
    x_temp = data_train_filled.iloc[:, :-1].as_matrix()  # 自变量
    y = data_train_filled.iloc[:, -1].as_matrix()  # 因变量

    '''Feature selection'''
    X, dropped_feature_name, len_feature_choose = gbdt_feature_selection(feature_name, x_temp, y, '0.1*mean')

    '''调参'''
    gbdt_model = GradientBoostingClassifier()

    param_grid = {'learning_rate': [0.01, 0.05, 0.1],
                  'n_estimators': [100, 150, 200, 250, 300, 400],
                  'subsample': [0.8, 1.0],
                  'min_samples_split': [2, 3],
                  'min_samples_leaf': [1, 2],
                  'max_depth': [3, 4, 5],
                  # 'random_state': [None, 0, 10, 100],
                  # 'max_features': [None, 'auto', 'log2']
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


def gbdt_without_cv_modeling():
    """

    :return:
    """

    '''Data input'''
    data_b_train = pd.read_csv('../data/B_train_final.csv', index_col='no')
    data_test = pd.read_csv('../data/B_test_final.csv', index_col='no')

    data_train = data_b_train

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
    X, dropped_feature_name, len_feature_choose = lgb_feature_selection(feature_name, x_temp, y, "0.1*mean")

    '''处理 验证集 B_test'''
    data_test_filled = data_test.fillna(value=10)
    data_test_filled_after_feature_selection = data_test_feature_drop(data_test_filled, dropped_feature_name)

    '''Choose a classification model'''
    parameter_n_estimators = 400
    classifier = GradientBoostingClassifier(n_estimators=parameter_n_estimators)

    a_model = classifier.fit(X, y)

    prob_of_b_test = a_model.predict_proba(data_test_filled_after_feature_selection)  # 对B_test进行预测

    result_file_name = '../result/B_test_gbdt_predict_without_cv_fillna_10_rd_' + str(s) + '_N_' + str(parameter_n_estimators) + '_features_' + \
                       str(len_feature_choose) + '.csv'

    write_predict_results_to_csv(result_file_name, data_test_user_id, prob_of_b_test[:, 1].tolist())
