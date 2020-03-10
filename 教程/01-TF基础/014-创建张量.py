from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import time
import tempfile
from tensorflow import keras


#创建张量
#从Numpy，List对象创建
#通过tf.convert_to_tensor 可以创建新Tensor，并将保存在Python List 对象或者NumpyArray 对象中的数据导入到新Tensor 中
#convert_to_tensor和constant作用一样
tf.convert_to_tensor([1,2.])
tf.convert_to_tensor(np.array([[1,2.],[3,4]]))

#创建全0，全1张量
tf.zeros([]),tf.ones([])
#创建全0，全1向量
tf.zeros([1]),tf.ones([1])
#创建全0矩阵，全1矩阵
tf.zeros([2,2]),tf.ones([2,3])
#创建与张量a形状一样的全1张量
a = tf.zeros([3,2])
tf.ones_like(a)

#创建自定义数值张量
#创建元素为-1的标量
tf.fill([],-1)
#创建所有元素为-1的向量
tf.fill([1],-1)
#创建所有元素为99的矩阵
tf.fill([2,2], 99)

#创建已知分布的张量
#创建均值为0，标准差为1（默认）的正态分布的张量
#中括号内有几个参数，生成的tensor就有几层括号
a = tf.random.normal([3,3])
#创建均值为1，标准差为2的正态分布的张量
tf.random.normal([2,2], mean=1,stddev=2)
#创建采样区间为[0,1]，shape为[2,2]的矩阵
tf.random.uniform([2,2])
#创建采样区间为[3,10]，shape为[2,2]的矩阵
tf.random.uniform([2,2],minval=3,maxval=10)
#如果需要均匀采样整形类型的数据，必须指定采样区间的最大值maxval 参数，同时制定数据类型为tf.int*型
tf.random.uniform([2,2],maxval=100,dtype=tf.int32)

#创建序列
#创建0-9，步长为1的整型序列
tf.range(10)
#创建0~9，步长为2的整型序列
tf.range(10,delta=2)
