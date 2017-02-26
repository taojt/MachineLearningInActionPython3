#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @filename:tree
# @Date    :2017-02-26
# @Author  :Vinicier
# @Blog    :http://www.taojt.xyz

from math import log
import operator


def calc_shannon_entropy(data_set):
    """
    Calculate Shannon Entropy
    :param data_set:  data set
    :return shannon_entropy: shannon entropy value
    """
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    shannon_entropy = 0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_entropy -= prob * log(prob, 2)
    return shannon_entropy


def create_data_set():
    """
    Create Initial Data Set
    :return:
    """
    data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def split_data_set(data_set, axis, value):
    """
    Split Data Set
    :param data_set: data set
    :param axis:  split data set with feature axis
    :param value: feature value
    :return:
    """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """
    Choose best feature to split with the biggest information gain
    :param data_set:
    :return:
    """
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_entropy(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]
        unique_vals = set(feat_list)
        # print("i : %d, unique values = %s" % (i, unique_vals))
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            # print("sub_data_set:%s" % sub_data_set)
            prob = len(sub_data_set) / float(len(data_set))

            new_entropy += prob * calc_shannon_entropy(sub_data_set)
        info_gain = base_entropy - new_entropy
        # print("info gain: %f" % info_gain)
        if (info_gain >= best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0] == len(class_list)):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feature = choose_best_feature_to_split(data_set)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label:{}}
    del(labels[best_feature])
    feature_values = [example[best_feature] for example in data_set]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(split_data_set(data_set,best_feature,value),sub_labels)
    return my_tree


# Test
if __name__ == '__main__':
    my_data, labels = create_data_set()
    print('my data: \t', my_data)
    shannon_entropy = calc_shannon_entropy(my_data)
    print('Shannon Entropy: \t', shannon_entropy)

    # 分类越多，熵越大，我们添加分类
    my_data[0][-1] = 'maybe'
    print('\nNew my data: \t', my_data)
    shannon_entropy = calc_shannon_entropy(my_data)
    print('The New Shannon Entropy: \t', shannon_entropy)

    # Split Data Set
    my_data, labels = create_data_set()
    ret_data_set_0 = split_data_set(my_data, 0, 0)
    ret_data_set_1 = split_data_set(my_data, 0, 1)
    print('\nAfter split with value is 0, the ret_data_Set is : \t %s\n' % ret_data_set_0)
    print('\nAfter split with value is 1, the ret_data_Set is : \t %s\n' % ret_data_set_1)

    # Choose Best Feature to Split
    my_data, labels = create_data_set()
    best_feature = choose_best_feature_to_split(my_data)
    print('my data: \t ', my_data)
    print('best feature :\t %d \n' % best_feature)

    #  create tree
    my_data,labels = create_data_set()
    my_tree = create_tree(my_data, labels)
    print('My Tree is : \t %s \n' % my_tree)
