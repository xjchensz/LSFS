import pandas as pd
import numpy as np
import scipy as sp
import sys
from fisher_score_FUN import *

def append_module_path():
    import sys
    paths = [ \
        "../gen_data",
        "../evaluate",
        "../read_data",
        "../PRPC"
    ]
    
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)
        
append_module_path()
import gen_data
import evaluate
import read_data
import PRPC_FUN


file_path = "..\\..\\data_selected\\gene\\brain\\"

selected_data_file_name = "selected_data"
# selected_feature_file_name = "selected_features"
selected_cluster_name_file_name = "selected_cluster_names"

unselected_data_file_name = "unselected_data"
# unselected_feature_file_name = "unselected_features"
unselected_cluster_name_file_name = "unselected_cluster_names"
example_rate = 50
feature_rate = 10

output_file_name = file_path + "prpc_result" + "_" +  str(example_rate) + "_" + str(feature_rate) + "" + ".txt"
XL_train, YL_train, XU_train, YU_train  = read_data.get_data(file_path, selected_data_file_name, selected_cluster_name_file_name,\
                        unselected_data_file_name, unselected_cluster_name_file_name, example_rate, feature_rate)


feature_order = fisher_score(XL_train, YL_train)

num_feature = len(feature_order)
if num_feature > 300:
    num_feature = 300
acc_array = evaluate.cal_many_acc(XL_train, YL_train, XU_train, YU_train,\
                           feature_order, num_feature = num_feature)


evaluate.plot_array_like(acc_array, xlabel_name="number feature", ylabel_name="accuracy")

