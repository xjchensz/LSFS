#!/usr/bin/python
# -*- coding:utf-8 -*- 

import pandas as pd
import scipy as sp
import numpy as np
import time


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


from my_math import *
from PRPC_FUN import *
from read_data import *
from evaluate import *

file_path = "..\\..\\data_selected\\gene\\brain\\"

selected_data_file_name = "selected_data"
# selected_feature_file_name = "selected_features"
selected_cluster_name_file_name = "selected_cluster_names"

unselected_data_file_name = "unselected_data"
# unselected_feature_file_name = "unselected_features"
unselected_cluster_name_file_name = "unselected_cluster_names"
example_rate = 50
feature_rate = 1

output_file_name = file_path + "prpc_result" + "_" +  str(example_rate) + "_" + str(feature_rate) + "" + ".txt"




XL_train, YL_train, XU_train, YU_train  = get_data(file_path, selected_data_file_name, selected_cluster_name_file_name,\
                        unselected_data_file_name, unselected_cluster_name_file_name, example_rate, feature_rate)



order_Fs, time_dual = prpc(XL_train, YL_train, XU_train, output_file_name="feature_order")


num_feature = len(order_Fs)
if num_feature > 300:
    num_feature = 300
    
acc_array = cal_many_acc(XL_train, YL_train, XU_train, YU_train,\
                           order_Fs, num_feature = num_feature)

print(order_Fs)
print("===================================================================")
print("===================================================================")
# print("accuracy : ", a)
print("time : ", time_dual)

plot_array_like(acc_array, xlabel_name="number feature", ylabel_name="accuracy")