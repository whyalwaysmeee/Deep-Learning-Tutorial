from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import time
import tempfile
from tensorflow import keras




#维度变换
#必要性：将维度不同、不能进行计算的张量变成可操作的张量
#reshape
"""
张量的视图就是我们理解张量的方式，比如shape 为[2,4,4,3]的张量A，我们从逻辑上可以理解
为2 张图片，每张图片4 行4 列，每个位置有RGB 3 个通道的数据；张量的存储体现在张
量在内存上保存为一段连续的内存区域，对于同样的存储，我们可以有不同的理解方式，
比如上述A，我们可以在不改变张量的存储下，将张量A 理解为2 个样本，每个样本的特
征为长度48 的向量。这就是存储与视图的关系

改变视图是神经网络中非常常见的操作，可以通过串联多个Reshape 操作来实现复杂
逻辑，但是在通过Reshape 改变视图时，必须始终记住张量的存储顺序，新视图的维度顺
序不能与存储顺序相悖，否则需要通过交换维度操作将存储顺序同步过来。举个例子，对
于shape 为[4,32,32,3]的图片数据，通过Reshape 操作将shape 调整为[4,1024,3]，此时视图
的维度顺序为𝑏 − 𝑝𝑖𝑥𝑒𝑙 − 𝑐，张量的存储顺序为[𝑏, ℎ, , 𝑐]。可以将[4,1024,3]恢复为:
❑ [𝑏, ℎ, w, 𝑐] = [4,32,32,3]时，新视图的维度顺序与存储顺序无冲突，可以恢复出无逻辑
问题的数据
❑ [𝑏, w, ℎ, 𝑐] = [4,32,32,3]时，新视图的维度顺序与存储顺序冲突
❑ [ℎ ∗ w ∗ 𝑐, 𝑏] = [3072,4]时，新视图的维度顺序与存储顺序冲突
"""
x = tf.range(96)
x = tf.reshape(x,[2,4,4,3])
#张量的维度数和形状
x.ndim,x.shape
#通过tf.reshape(x, new_shape)，可以将张量的视图任意的合法改变
tf.reshape(x,[2,-1])
tf.reshape(x,[2,4,12])
tf.reshape(x,[2,-1,3])

#增删维度
#增加维度
"""
增加一个长度为1 的维度相当于给原有的数据增加一个新维度的概念，维度
长度为1，故数据并不需要改变，仅仅是改变数据的理解方式，因此它其实可以理解为改
变视图的一种特殊方式
"""
x = tf.random.uniform([28,28],maxval=10,dtype=tf.int32)
#通过tf.expand_dims(x, axis)可在指定的axis 轴前可以插入一个新的维度
x = tf.expand_dims(x,axis=2)
#删除维度
#如果不指定维度参数axis，即tf.squeeze(x)，那么会默认删除所有长度为1 的维度
x = tf.squeeze(x, axis=0)

#交换维度
#[𝑏, ℎ, w, 𝑐]转换到[𝑏, 𝑐, ℎ, w]
#图片张量shape 为[2,32,32,3]，图片数量、行、列、通道数的维度索引分别为0,1,2,3
x = tf.random.normal([2,32,32,3])
#新维度的排序为图片数量、通道数、行、列，对应的索引号为[0,3,1,2]
tf.transpose(x,perm=[0,3,1,2])
#通过tf.transpose 完成维度交换后，张量的存储顺序已经改变，视图也随之改变，后续的所有操作必须基于新的存续顺序进行

#数据复制
b = tf.constant([1,2])
#增加维度
b = tf.expand_dims(b, axis=0)
#在batch维度上复制数据1份
b = tf.tile(b, multiples=[2,1])
"""
结果如下：
tf.Tensor([1 2], shape=(2,), dtype=int32)
tf.Tensor([[1 2]], shape=(1, 2), dtype=int32)
tf.Tensor(
[[1 2]
 [1 2]], shape=(2, 2), dtype=int32)
可以看到shape由1变为2
"""

