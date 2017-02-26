#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @filename:tree_plotter
# @Date    :2017-02-26
# @Author  :Vinicier
# @Blog    :http://www.taojt.xyz
from matplotlib import pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                             textcoords='axes fraction', va='center', ha='center', bbox=node_type,
                             arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node("决策节点", (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node("叶子节点", (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


# test
if __name__ == '__main__':
    create_plot()
