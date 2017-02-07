#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @filename:test
# @Date    :2017-01-20
# @Author  :Vinicier
# @Blog    :http://www.taojt.xyz
from numpy import *
import operator
import logging
from __init__ import *
log = logging.getLogger('knn')


def create_dataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDisIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDisIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file_to_matrix(filename):
    with open(filename) as fr:
        array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)
    return_mat = zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(list_from_line[-1])
        index += 1
    return return_mat, class_label_vector

def auto_norm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals,(m,1))
    norm_data_set = norm_data_set/tile(ranges,(m,1))
    return norm_data_set, ranges, min_vals


if __name__ == '__main__':
    group, labels = create_dataset()
    result = classify0([0, 0], group, labels, 3)
    log.debug('The KNN classify result is : %s ' % result)
    dating_data_mat, class_label_vector = file_to_matrix('datingTestSet.txt')
    log.debug(dating_data_mat)
    norm_data_set = auto_norm(dating_data_mat)
    log.debug(norm_data_set)
