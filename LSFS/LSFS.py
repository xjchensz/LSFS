import pandas as pd
import numpy as np
import scipy as sp
import os
import random
import time
import sys

from LSFS_FUN import *
from LSFS_TEST import *
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

output_file_name = file_path + "lsfs_result" + "_" +  str(example_rate) + "_" + str(feature_rate) + "" + ".txt"

XL_train, YL_train, XU_train, YU_train  = get_data(file_path, selected_data_file_name, selected_cluster_name_file_name, \
                                    unselected_data_file_name, unselected_cluster_name_file_name, example_rate, feature_rate)

# feature_order, time_dual, a = run_accuracy(lsfs, XL_train,YL_train,XU_train,YU_train, 10, output_file_name)

XL, YL, XU, YU = XL_train.copy(), YL_train.copy(), XU_train.copy(), YU_train.copy()
YL = read_data.label_n1_to_nc(YL)
YU = read_data.label_n1_to_nc(YU)
feature_order, time_dual = lsfs(XL, YL, XU, output_file_name="feature_order")

num_feature = len(feature_order)
if num_feature > 300:
    num_feature = 300

acc_array = cal_many_acc(XL_train, YL_train, XU_train, YU_train,\
                           feature_order, num_feature = num_feature)


print(feature_order)
print("===================================================================")
print("===================================================================")
# print("accuracy : ", a)
print("time : ", time_dual)


plot_array_like(acc_array, xlabel_name="number feature", ylabel_name="accuracy")
