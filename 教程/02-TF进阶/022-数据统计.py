import  matplotlib
from 	matplotlib import pyplot as plt
import  tensorflow as tf
from   keras import datasets
import  os
import numpy as np
from tensorflow import keras

#范数
x = tf.ones([2,2,2])
#L1范数
tf.norm(x,ord=1)
#L2范数
tf.norm(x,ord=2)
#无穷范数
tf.norm(x,ord=np.inf)

#最大最小值、均值、和
#不指定维度参数时求全局
#最大值
x = tf.random.normal([4,10])
#计算这个4行10列的矩阵的每一行，即第2个维度上的最大值
tf.reduce_max(x,axis=1)
#计算这个4行10列的矩阵的每一行，即第2个维度上的最小值
tf.reduce_min(x,axis=1)
#计算这个4行10列的矩阵的每一行，即第2个维度上的均值
tf.reduce_mean(x,axis=1)
#计算这个4行10列的矩阵的每一行，即第2个维度上的和
tf.reduce_sum(x,axis=-1) # 求和
#平均误差MSE
out = tf.random.normal([4,10]) # 网络预测输出
y = tf.constant([1,2,2,0]) # 真实标签
y = tf.one_hot(y,depth=10) # one-hot 编码
loss = keras.losses.mse(y,out) # 计算每个样本的误差
loss = tf.reduce_mean(loss) # 平均误差
#在分类任务中，除了要获取最值信息，还希望获得最值所在的索引号
pred = tf.argmax(out, axis=1) # 选取概率最大的位置
pred = tf.argmin(out, axis=1) # 选取概率最小的位置
















