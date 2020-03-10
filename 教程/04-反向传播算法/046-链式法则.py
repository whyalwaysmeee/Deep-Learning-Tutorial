import  matplotlib
from 	matplotlib import pyplot as plt
import  tensorflow as tf
from   keras import datasets
import  os
from tensorflow import keras
from keras import layers
import pandas as pd
from tensorflow import losses
import numpy as np

# 构建待优化变量
x = tf.constant(1.)
w1 = tf.constant(2.)
b1 = tf.constant(1.)
w2 = tf.constant(2.)
b2 = tf.constant(1.)
# 构建梯度记录器
with tf.GradientTape(persistent=True) as tape:
# 非tf.Variable 类型的张量需要人为设置记录梯度信息
  tape.watch([w1, b1, w2, b2])
# 构建2 层线性网络
  y1 = x * w1 + b1
  y2 = y1 * w2 + b2
# 独立求解出各个偏导数
dy2_dy1 = tape.gradient(y2, [y1])[0]
dy1_dw1 = tape.gradient(y1, [w1])[0]
dy2_dw1 = tape.gradient(y2, [w1])[0]
# 验证链式法则
print(dy2_dy1 * dy1_dw1)
print(dy2_dw1)
























