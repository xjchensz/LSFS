#!usr/bin/python
# -*- coding:utf-8 *-

import pandas as pd
import scipy as sp
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt


#获取脚本文件的当前路径
def cur_file_dir():
    #获取脚本路径
    path = sys.path[0]
    #判断为脚本文件还是py2exe编译后的文件，如果是脚本文件，则返回的是脚本的目录，如果是py2exe编译后的文件，则返回的是编译后的文件路径
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)
    
    
    
# 平方和
def _sum_of_squares(a, axis=0):
    """
    Squares each element of the input array, and returns the sum(s) of that.
    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which to calculate. Default is 0. If None, compute over
        the whole array `a`.
    Returns
    -------
    sum_of_squares : ndarray
        The sum along the given axis for (a**2).
    See also
    --------
    _square_of_sums : The square(s) of the sum(s) (the opposite of
    `_sum_of_squares`).
    """
    a, axis = _chk_asarray(a, axis)
    return np.sum(a*a, axis)



def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)

    return a, outaxis



# 两向量的皮尔逊系数 
def compute_pearson(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(_sum_of_squares(xm) * _sum_of_squares(ym))
    r = r_num / r_den
    return r



def prpc(XL, YL, XU, output_file_name="feature_order"):
    # 特征的索引号
    F = set(range(XL.shape[1]))
    Fs = set()
    order_Fs = []
    Fa = F - Fs

    # 选取的特征个数
    S = len(F)

    ln = XL.shape[0]
    X = sp.concatenate((XL, XU), axis = 0)

    s = time.clock()
    for k in range(1,S+1):
        max_pearson = float("-inf")
        Fk = -1

        for Fj in Fa:

            pearson1 = compute_pearson(X[:ln,Fj], YL)

    #         print(label_data[:,Fj])
    #         print(YL)
    #         print(pearson1)
    #         print(Fj)


            pearson2_sum = 0

            for Fi in Fs:
                pearson2 = compute_pearson(X[:,Fj], X[:,Fi])
                pearson2_sum += pearson2


            if k == 1:
                pearson2_sum = 0
            elif k > 1:
                pearson2_sum /= k - 1

    #         print(pearson1, pearson2_sum)

            pearson1 -= pearson2_sum

            if pearson1 > max_pearson:
                max_pearson = pearson1
                Fk = Fj

        Fs = Fs | {Fk}
        Fa = Fa - {Fk}

        order_Fs.append(Fk)

    time_dual = time.clock() - s

    with open(output_file_name, "w+") as result_file:
        print("\n".join([str(w) for w in order_Fs]), file=result_file)
    
    return order_Fs, time_dual









