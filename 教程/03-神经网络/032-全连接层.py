import  matplotlib
from 	matplotlib import pyplot as plt
import  tensorflow as tf
from   keras import datasets
import  os
from tensorflow import keras
from keras import layers

#张量方式实现
x = tf.random.normal([2,784])   #输入
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))   #权值矩阵
b1 = tf.Variable(tf.zeros([256]))  #偏置
o1 = tf.matmul(x,w1) + b1 # 线性变换
o1 = tf.nn.relu(o1) # 激活函数

#层方式实现
x = tf.random.normal([4,28*28])
# 创建全连接层，指定输出节点数和激活函数
fc = layers.Dense(512, activation=tf.nn.relu)
h1 = fc(x) # 通过fc 类完成一次全连接层的计算
fc.kernel # 获取Dense 类的权值矩阵
fc.bias # 获取Dense 类的偏置向量
fc.trainable_variables  # 返回待优化参数列表
fc.variables # 返回所有参数列表
fc.non_trainable_variables  #返回所有不需要优化的参数列表



















