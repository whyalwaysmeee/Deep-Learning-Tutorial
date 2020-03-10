from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import time
import tempfile
from tensorflow import keras



#张量的典型应用
#向量
#偏置张量b可用向量来表示。考虑 2 个输出节点的网络层，我们创建长度为2 的偏置向量𝒃，并累加在每个输出节点上
#z=wx,模拟获得激活函数的输入z
z = tf.random.normal([4,2])
#模拟偏置向量
b = tf.zeros([2])
#累加偏置
z = z + b
"""
通过高层接口类Dense()方式创建的网络层，张量W 和𝒃存储在类的内部，由类自动创
建并管理。可以通过全连接层的bias 成员变量查看偏置变量𝒃，例如创建输入节点数为4，
输出节点数为3 的线性层网络，那么它的偏置向量b 的长度应为3
"""
fc = keras.layers.Dense(3) # 创建一层Wx+b，输出节点为3
#通过build 函数创建W,b 张量，输入节点为4
fc.build(input_shape=(2,4))
#查看偏置
fc.bias

#矩阵
#创建一个2个4维元素的矩阵
x = tf.random.normal([2,4])
#设置权重
w = tf.ones([4,3])
#设置偏置
b = tf.zeros([3])
#计算结果
o = x@w + b

#3维张量
"""
三维的张量一个典型应用是表示序列信号，它的格式是
𝑋 = [𝑏, 𝑠𝑒𝑞𝑢𝑒𝑛𝑐𝑒 𝑙𝑒𝑛, 𝑓𝑒𝑎𝑡𝑢𝑟𝑒 𝑙𝑒𝑛]
其中𝑏表示序列信号的数量，sequence len 表示序列信号在时间维度上的采样点数，feature
len 表示每个点的特征长度
"""
#自动加载IMDB 电影评价数据集
(x_train,y_train),(x_test,y_test)=keras.datasets.imdb.load_data(num_words=10000)
#将句子填充、截断为等长80 个单词的句子
x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen=80)
x_train.shape
"""
可以看到x_train 张量的shape 为[25000,80]，其中25000 表示句子个数，80 表示每个句子
共80 个单词，每个单词使用数字编码方式。我们通过layers.Embedding 层将数字编码的单
词转换为长度为100 个词向量：
"""
#创建词向量Embedding 层类
embedding = keras.layers.Embedding(10000, 100)
#将数字编码的单词转换为词向量
out = embedding(x_train)
#查看
out.shape
#结果为：TensorShape([25000, 80, 100])
#可以看到，经过Embedding 层编码后，句子张量的shape 变为[25000,80,100]，其中100表示每个单词编码为长度100的向量。

#4维张量
"""
4维张量在卷积神经网络中应用的非常广泛，它用于保存特征图(Feature maps)数据，
格式一般定义为
[𝑏, ℎ, w, 𝑐]
其中𝑏表示输入的数量，h/w分布表示特征图的高宽，𝑐表示特征图的通道数，部分深度学
习框架也会使用[𝑏, 𝑐, ℎ, w]格式的特征图张量，例如PyTorch。图片数据是特征图的一种，
对于含有RGB 3 个通道的彩色图片，每张图片包含了h 行w 列像素点，每个点需要3 个数
值表示RGB 通道的颜色强度，因此一张图片可以表示为[h, w, 3]
"""
#神经网络中一般并行计算多个输入以提高计算效率，故𝑏张图片的张量可表示为[𝑏, ℎ, w, 3]
# 创建32x32 的彩色图片输入，个数为4
x = tf.random.normal([4,32,32,3])
# 创建卷积神经网络
layer = keras.layers.Conv2D(16,kernel_size=3)
out = layer(x) # 前向计算
out.shape # 输出大小
#结果为：TensorShape([4, 30, 30, 16])
#其中卷积核张量也是4 维张量，可以通过kernel 成员变量访问：
keras.layer.kernel.shape
#结果为：TensorShape([3, 3, 3, 16])
