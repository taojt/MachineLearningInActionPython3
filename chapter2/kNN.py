#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @filename:test
# @Date    :2017-01-20
# @Author  :Vinicier
# @Blog    :http://www.taojt.xyz
from numpy import *
import operator


def createDataSet():
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
        return_mat[index,:] = list_from_line[0:3]
        class_label_vector.append(list_from_line[-1])
        index += 1
    return return_mat, class_label_vector





if __name__ == '__main__':
    group, labels = createDataSet()
    result = classify0([0, 0], group, labels, 3)
    print('The kNN classify result is : ', result)
    dating_data_mat, class_label_vector = file_to_matrix('datingTestSet.txt')
    print(dating_data_mat)