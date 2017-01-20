#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @filename:test
# @Date    :2017-01-20
# @Author  :Vinicier
# @Blog    :http://www.taojt.xyz
from numpy import *
import operator
def createDataSet():
    group = array([[1.0,1.1], [1.0, 1.0], [0, 0],[0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

if __name__ == '__main__':
    createDataSet()

