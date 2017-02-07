#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @filename:__init__
# @Date    :2017-02-07
# @Author  :Vinicier
# @Blog    :http://www.taojt.xyz
from numpy import array
# logging 模块增加程序日志功能
import logging
import sys

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] \t %(levelname)s : \t %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    filemode='w',
                    stream=sys.stdout)
__all__ = ['logging', 'array']
