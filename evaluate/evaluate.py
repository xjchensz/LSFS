#!/usr/bin/python
# -*- coding:utf-8 -*- 

import pandas as pd
import scipy as sp
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def accuracy(true_label, clust_label):
    """
    true_label : 想要的结果
    clust_label : 算法的预测
    """
    tp = tn = 0
    fp = fn = 0
    for i in range( len(true_label) ):
        for j in range( len(clust_label) ):
            # true positive
            if (true_label[i] == true_label[j] and clust_label[i] == clust_label[j]):
                tp+=1
            # true negative 
            elif (true_label[i] != true_label[j] and clust_label[i] != clust_label[j]):
                tn+=1
            # false positive
            elif (true_label[i] != true_label[j] and clust_label[i] == clust_label[j]):
                fp+=1
            else:
            # false negative
                fn+=1
    return (tp+tn)*1.0/(tp+tn+fp+fn)


def select_data(XL, YL, XU, YU, feature_order, sel_num = None):
    """
    重组和筛选特征
    """
    if sel_num == None:
        sel_num = len(feature_order)
        
    X = sp.concatenate((XL, XU), axis = 0)
    sel_feature = feature_order[:sel_num]
    X = X[:,sel_feature]
    Y = sp.concatenate((YL, YU), axis = 0)
    return X, Y


def get_train_test_rate(x, y, rate=None):
    """
    抽样=》训练，测试
    x 特征 （样本个数 x 特征个数）
    y 类别
    """
    n_samples = len(x)
    if rate == None:
        train_mask = np.random.randint(0, 2, size=n_samples).astype(np.bool)
        x_train, y_train = x[train_mask, :], y[train_mask]
        x_test, y_test = x[~train_mask, :], y[~train_mask]
    else:
        row_selected_rate = rate
        row_selected = random.sample(range(n_samples), int(row_selected_rate*n_samples))
        row_unselected = list(set(range(n_samples)) - set(row_selected))
        
        x_train, y_train = x[row_selected, :], y[row_selected]
        x_test, y_test = x[row_unselected, :], y[row_unselected]
        
    return x_train, y_train, x_test, y_test


def run_acc(x, y, row_selected_rate=None):
    cnt = 100
    sum_accuracy = 0.0
    avg_accuracy = 0.0
    for i in range(cnt):
        row_num = x.shape[0]
        # 根据比例筛选数据
        row_selected_rate = 0.5
#         row_selected = random.sample(range(row_num), int(row_selected_rate*row_num))
#         row_unselected = list(set(range(row_num)) - set(row_selected))
        
#         train_set = X[row_selected, :]
#         train_label = Y[row_selected]
#         test_set = X[row_unselected, :]
#         test_label = Y[row_unselected]
        
        train_set, train_label, test_set, test_label = get_train_test_rate(x, y, rate=row_selected_rate)
        
        
        # fit a SVM model to the data
        model = svm.LinearSVC(loss='hinge')
        model.fit(train_set, train_label)
        # print(model)
        # make predictions
        expected = test_label
        predicted = model.predict(test_set)
        
        sum_accuracy += accuracy(expected, predicted)
    avg_accuracy = sum_accuracy / cnt
    return avg_accuracy



def cal_many_acc(XL_train, YL_train, XU_train, YU_train,\
                           feature_order, num_feature = 10):
    acc_array = np.zeros(num_feature)
    for i in range(1,num_feature):
        X,Y = select_data(XL_train, YL_train, XU_train, YU_train,\
                           feature_order, sel_num=i)
        a = run_acc(X,Y)
        acc_array[i] = a
#        print(a)
    return acc_array



def plot_array_like(array_1d, xlabel_name="number feature", ylabel_name="accuracy"):
    figsize = (8, 5)
    fig = plt.figure(figsize=figsize)

    plt.plot(range(len(array_1d)), array_1d)

    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)

    plt.xlim(0, len(array_1d))
    # plt.ylim(0,1)
    plt.show()
