#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @filename:matplotlib_test
# @Date    :2017-02-07
# @Author  :Vinicier
# @Blog    :http://www.taojt.xyz

import matplotlib
import matplotlib.pyplot as plt
from  __init__ import *

fig = plt.figure()
ax = fig.add_subplot(111)
# 导入数据文件
from knn import file_to_matrix

dating_data_mat, class_label_vector = file_to_matrix('datingTestSet.txt')
ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2])
# ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2],15.0 * array(class_label_vector), 15.0 * array(class_label_vector) )
plt.show()
