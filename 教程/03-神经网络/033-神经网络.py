import  matplotlib
from 	matplotlib import pyplot as plt
import  tensorflow as tf
from   keras import datasets
import  os
from tensorflow import keras
from keras import layers

"""
现在考虑实现一个4层神经网络：
输入层：[b,784]
隐藏层1：[256]
隐藏层2：[128]
隐藏层3：[64]
输出层：[b,10]
"""
#张量方式实现
x = tf.random.normal([4,28*28])
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
# 隐藏层2 张量
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
# 隐藏层3 张量
w3 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))
# 输出层张量
w4 = tf.Variable(tf.random.truncated_normal([64, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))
"""
在使用 TensorFlow 自动求导功能计算梯度时，需要将前向计算过程放置在
tf.GradientTape()环境中，从而利用GradientTape 对象的gradient()方法自动求解参数的梯
度，并利用optimizers 对象更新参数
"""
with tf.GradientTape() as tape: # 梯度记录器
# x: [b, 28*28]
# 隐藏层1 前向计算，[b, 28*28] => [b, 256]
   h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
   h1 = tf.nn.relu(h1)
# 隐藏层2 前向计算，[b, 256] => [b, 128]
   h2 = h1@w2 + b2
   h2 = tf.nn.relu(h2)
# 隐藏层3 前向计算，[b, 128] => [b, 64]
   h3 = h2@w3 + b3
   h3 = tf.nn.relu(h3)
# 输出层前向计算，[b, 64] => [b, 10]
   h4 = h3@w4 + b4

#层方式实现
x = tf.random.normal([4,28*28])
fc1 = layers.Dense(256, activation=tf.nn.relu) # 隐藏层1
fc2 = layers.Dense(128, activation=tf.nn.relu) # 隐藏层2
fc3 = layers.Dense(64, activation=tf.nn.relu) # 隐藏层3
fc4 = layers.Dense(10, activation=None) # 输出层

x = tf.random.normal([4,28*28])
h1 = fc1(x) # 通过隐藏层1 得到输出
h2 = fc2(h1) # 通过隐藏层2 得到输出
h3 = fc3(h2) # 通过隐藏层3 得到输出
h4 = fc4(h3) # 通过输出层得到网络输出

model = keras.layers.sequential([layers.Dense(256, activation=tf.nn.relu) , # 创建隐藏层1
layers.Dense(128, activation=tf.nn.relu) , # 创建隐藏层2
layers.Dense(64, activation=tf.nn.relu) , # 创建隐藏层3
layers.Dense(10, activation=None) , # 创建输出层
])
out = model(x) # 前向计算得到输出
















