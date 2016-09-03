#!usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import scipy as sp
import sys



"""
Fr = sum n_i*(u_i -u)^2  /  sum n_i * theta_i^2
     i=1:c                  i=1:c
"""
def fisher_score(x_train, y_train):
    
    """
    特征
    """
    f_len = x_train.shape[1]
    x_f = set(range(f_len))
    
    """
    类别
    """
    y_c = set(y_train)
    
    feature_value = np.zeros(f_len, dtype=np.float)
    # 对于每个特征
    for f_i in x_f:
        # i特征的数据
        x_fi = x_train[:,f_i:f_i+1]

        """
        i特征的均值和方差
        """
        fi_mean = np.mean(x_fi)
        fi_var = np.var(x_fi)

        # 分子
        sum_1 = 0.0
        # 分母
        sum_2 = 0.0

        # 对于每个特征的每个类别
        for c_j in y_c:
            y_cj = y_train == c_j
            n_fi_cj = np.sum(y_cj)
            """
            i特征j类别的均值和方差
            """
            fi_cj_mean = np.mean(x_fi[y_cj, :], axis = 0)
            fi_cj_var = np.var(x_fi[y_cj, :], axis = 0)

            sum_1 += n_fi_cj  * ( sum(fi_cj_mean - fi_mean)**2 )
            sum_2 += n_fi_cj  * ( fi_cj_var**2 )

            
        if sum_2 == 0:
            feature_value[f_i] = 0.0
        else:
            feature_value[f_i] = sum_1/sum_2
        
    feature_order = np.argsort(feature_value)
    feature_order = feature_order[::-1]
    return feature_order