from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import time
import tempfile
from tensorflow import keras

#Broadcasting
x = tf.random.normal([2,4])
w = tf.random.normal([4,3])
b = tf.random.normal([3])
y = x@w+b
#x@w的结果的shape为[2,3]，b的shape为[3]
#上述加法并没有发生逻辑错误，那么它是怎么实现的呢？这是因为它自动调用Broadcasting函数tf.broadcast_to(x, new_shape)，将2者shape 扩张为相同的[2,3]，即上式可以等效为：
y = x@w + tf.broadcast_to(b,[2,3])
#操作符+在遇到shape不一致的2个张量时，会自动考虑将2个张量Broadcasting到一致的shape，然后再调用tf.add 完成张量相加运算
"""
Broadcasting 机制的核心思想是普适性，即同一份数据能普遍适合于其他位置。在验证
普适性之前，需要将张量shape 靠右对齐，然后进行普适性判断：对于长度为1 的维度，
默认这个数据普遍适合于当前维度的其他位置；对于不存在的维度，则在增加新维度后默
认当前数据也是普适性于新维度的，从而可以扩展为更多维度数、其他长度的张量形状
"""
a = tf.random.normal([32,1])
tf.broadcast_to(a,[2,32,32,3])
#利用broadcasting机制进行各种运算
a = tf.random.normal([32,1])
b = tf.broadcast_to(a,[2,32,32,3])
a+b,a-b,a*b,a/b