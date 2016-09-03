#!usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import scipy as sp
import os
import random
import time
import sys

def append_module_path():
    import sys
    paths = [ \
        "../gen_data",
        "../evaluate",
        "../read_data"
    ]
    
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)

append_module_path()
import gen_data
import evaluate
import read_data


def test_H():
    """
    expected
    array([[ 0.66666667, -0.33333333, -0.33333333],
       [-0.33333333,  0.66666667, -0.33333333],
       [-0.33333333, -0.33333333,  0.66666667]])
    """
    return compute_H(3)



def test_norm_2_1():
    """
    expected 4.2426406871192857
    """
    W = np.array([[1,1],[2,2]])
    return norm_2_1(W)



def test_Q():
    """
    (np.sqrt(2) +  np.sqrt(8)) / [np.sqrt(2), np.sqrt(8)]
    expected [[ 3. ,  0. ],
              [ 0. ,  1.5]]
    """
    W = np.array([[1,1],[2,2]])
    return compute_Q(W)



def print_W(W):
    with open("W.txt", "a+") as f:
        for w in W:
            print(w, file=f)
        print("\n========================\n", file=f)
        


def run_accuracy(fun, XL_train,YL_train,XU_train,YU_train, sel_num=5, output_file_name="feature_order"):
    XL, YL, XU, YU = XL_train.copy(), YL_train.copy(), XU_train.copy(), YU_train.copy()
    
    if fun.__name__.lower() == "lsfs":
        YL = read_data.label_n1_to_nc(YL)
        YU = read_data.label_n1_to_nc(YU)
    
    feature_order, time_dual = fun(XL, YL, XU, output_file_name=output_file_name)
    
    X,Y = evaluate.select_data(XL_train, YL_train, XU_train, YU_train,\
                           feature_order, sel_num=sel_num)
    a = evaluate.run_acc(X,Y)
    print("accuracy", ":", a)
    return feature_order, time_dual, a