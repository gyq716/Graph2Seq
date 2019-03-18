#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:58:55 2019

@author: heng
"""


import numpy as np
import torch

frel = open('/home/heng/python_py/encodingText2.txt','r')
fobj = open('/home/heng/python_py/word_split2.txt','r')
relLines = frel.readlines()
objLines = fobj.readlines()
rels = []
objs = []
dics = []

# 将每一行根据两个空格割分，用于根据关系来生成adjacency matrix
# error log:之前文本处理成了一个空格，split时会和relationship混淆
for rel in relLines:
    rels.append(rel[:-3].split('  '))
# 将每一行根据两个空格割分，用于给结点标号（生成字典）
for obj in objLines:
    objs.append(obj[:-1].split('  '))

# 生成字典用于标号
for obj in objs:
    i = 0
    dic = {}
    for obj_ in obj:
        dic[obj_] = i
        i += 1
    dics.append(dic)

matrixs = []
for i in range(len(dics)):   # len(dics) 行，dics是list
    dic = dics[i]   # dic是字典
    rel = rels[i]   # rel是list，存的是relationship，给adj matrix改1
    #print(rel)
    length = len(dic)
    matrix = np.zeros((length,length),dtype = int)
    for r in rel:
        rs = r.split(',')
        if len(rs) == 1:     # 只有单独object，无att和relationship
            if rs[0] in dic.keys():
                loc = dic[rs[0]]
                if loc >= length:   # 处理数据异常时
                    pass
                else:
                    matrix[loc][loc] = 1
            else:
                pass
        if len(rs) == 2:
            if rs[0] in dic.keys() and rs[1] in dic.keys():
                loc1 = dic[rs[0]]    # 有attribute或者relationship
                loc2 = dic[rs[1]]    # 处理成对称矩阵（无向）
                if loc1 >= length or loc2 >= length:  # 处理数据异常时
                    pass
                else:
                    matrix[loc1][loc2] = 1
                    matrix[loc2][loc1] = 1
            else:
                pass
    print(matrix)   
    
    
    
    
    
    