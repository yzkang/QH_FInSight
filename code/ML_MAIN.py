#!/usr/bin/python
#  -*- coding: utf-8 -*-
# date: 2017
# author: Kang Yan Zhe
# desc: New

from B_LGB_GBDT import gbdt_cv_modeling
from B_LGB_GBDT import gbdt_without_cv_modeling
from A_TL_GBDT_GBDT import without_cv_transfer_a_to_b_modeling
# from A_TL_LGB_LGB import without_cv_transfer_a_to_b_modeling

if __name__ == '__main__':

    '''对B_train的训练'''
    gbdt_cv_modeling()
    gbdt_without_cv_modeling()

    '''对A_train的迁移学习，使用GBDT+GBDT建模'''
    without_cv_transfer_a_to_b_modeling()









