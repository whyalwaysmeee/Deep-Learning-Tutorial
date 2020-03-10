from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import time
import tempfile
from tensorflow import keras

#待优化张量
#待优化张量用于支持梯度信息，是一种特殊的张量，用tf.Variable来包装
#例如在在训练神经网络的过程中，权值矩阵和偏置需要不断的改变，就需要使用Variable作为数据结构
#可以通过Variable来转化普通张量来得到
a = tf.constant([-1, 0, 1, 2])
aa = tf.Variable(a)
"""
其中张量的name 和trainable 属性是Variable 特有的属性，name 属性用于命名计算图中的
变量，这套命名体系是TensorFlow 内部维护的，一般不需要用户关注name 属性；trainable
表征当前张量是否需要被优化，创建Variable 对象是默认启用优化标志，可以设置
trainable=False 来设置张量不需要优化
"""
#也可以直接创建
a = tf.Variable([[1,2],[3,4]])
