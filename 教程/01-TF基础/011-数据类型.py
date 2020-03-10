from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import time
import tempfile
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.utils import plot_model
import graphviz
import pydot


#数值类型
#数值类型的张量是TensorFlow 的主要数据载体
"""
张量(Tensor) 所有维度数dim > 2的数组统称为张量。张量的每个维度也做轴(Axis)，
一般维度代表了具体的物理含义，比如Shape 为[2,32,32,3]的张量共有4 维，如果表
示图片数据的话，每个维度/轴代表的含义分别是：图片数量、图片高度、图片宽度、
图片通道数，其中2 代表了2 张图片，32 代表了高宽均为32，3 代表了RGB 3 个通
道。张量的维度数以及每个维度所代表的具体物理含义需要由用户自行定义
"""
#创建标量
a = tf.constant(33.3)
#查看类型
type(a)
#验证
tf.is_tensor(a)
#打印相关信息
print(a)

#创建向量
#必须通过List传给tf
#创建1个元素的向量
aa = tf.constant([1.2])
#创建多个元素的向量
bb = tf.constant([1,2,3.])
#创建矩阵
cc = tf.constant([[1,2],[3,4]])
#创建3维张量
dd = tf.constant([[[1,2],[3,4]],[[5,6],[7,8]]])

#字符串和布尔型张量使用较少






