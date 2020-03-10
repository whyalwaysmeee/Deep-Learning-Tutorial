from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import time
import tempfile
from tensorflow import keras


#数学运算
#加减乘除
"""
加减乘除是最基本的数学运算，分别通过tf.add, tf.subtract, tf.multiply, tf.divide 函数实
现，TensorFlow 已经重载了+ −∗/运算符，一般推荐直接使用运算符来完成加减乘除运
算。
整除和余除也是常见的运算之一，分别通过//和%运算符实现
"""

#乘方
x = tf.range(4)
tf.pow(x,3)
x ** 2
#开方
x ** (0.5)
#平方
tf.square(x)
#平方根
tf.sqrt(x)

#指数、对数
x = tf.constant([1.,2.,3.])
2**x
tf.pow(2,x)
#自然指数
tf.exp(1.)
#对数
tf.math.log(x)

#矩阵相乘
"""
a和b能够矩阵相乘的条件是，a的倒数第一个维度长度(列)和
b的倒数第二个维度长度(行)必须相等。比如张量a: shape:[4,3,28,32]可以与张量b:
shape:[4,3,32,2]进行矩阵相乘
"""
a = tf.random.normal([4,3,23,32])
b = tf.random.normal([4,3,32,2])
a@b
#或者
tf.matmul(a,b)
