from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import time
import tempfile
from tensorflow import keras

#数值精度
"""
对于数值类型的张量，可以保持为不同字节长度的精度，如浮点数3.14 既可以保存为
16-bit 长度，也可以保存为32-bit 甚至64-bit 的精度。Bit 位越长，精度越高，同时占用的
内存空间也就越大。常用的精度类型有tf.int16, tf.int32, tf.int64, tf.float16, tf.float32,
tf.float64，其中tf.float64 即为tf.double
"""
#创建张量时可以指定精度
tf.constant(123456789, dtype=tf.int16)
tf.constant(123456789, dtype=tf.int32)
"""
对于大部分深度学习算法，一般使用tf.int32, tf.float32 可满足运算精度要求，部分对
精度要求较高的算法，如强化学习，可以选择使用tf.int64, tf.float64 精度保存张量。
"""

#类型转换
#用cast，将float转化为double型
a = tf.constant(np.pi, dtype=tf.float16)
tf.cast(a, tf.double)
#布尔型与整型之间的转换
#一般默认0 表示False，1 表示True，在TensorFlow 中，将非0 数字都视为True
a = tf.constant([True, False])
tf.cast(a,tf.int32)
a = tf.constant([-1, 0, 1, 2])
tf.cast(a, tf.bool)

