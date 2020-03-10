import  matplotlib
from 	matplotlib import pyplot as plt
import  tensorflow as tf
from   keras import datasets
import  os
from tensorflow import keras
from keras import layers

"""
常见的误差计算函数有均方差、交叉熵、KL 散度、Hinge Loss 函数等，其中均方差函
数和交叉熵函数在深度学习中比较常见，均方差主要用于回归问题，交叉熵主要用于分类
问题
"""

#均方差
"""
均方差误差(Mean Squared Error, MSE)函数把输出向量和真实向量映射到笛卡尔坐标系
的两个点上，通过计算这两个点之间的欧式距离(准确地说是欧式距离的平方)来衡量两个
向量之间的差距
"""
o = tf.random.normal([2,10]) # 构造网络输出
y_onehot = tf.constant([1,3]) # 构造真实值
y_onehot = tf.one_hot(y_onehot, depth=10)
loss = keras.losses.MSE(y_onehot, o) # 计算均方差
loss = tf.reduce_mean(loss) # 计算batch均方差，即每个样本的平均方差
#通过层方式实现，对应的类为keras.losses.MeanSquaredError()：
# 创建MSE 类
criteon = keras.losses.MeanSquaredError()
loss = criteon(y_onehot,o) # 计算batch 均方差

#交叉熵
"""
通过变换，交叉熵可以分解为p 的熵𝐻(𝑝)与p,q 的KL 散度(Kullback-Leibler Divergence)的
和：
          𝐻(𝑝, 𝑞) = 𝐻(𝑝) + 𝐷𝐾𝐿(𝑝|𝑞)
其中 KL 定义为
          𝐷𝐾𝐿(𝑝|𝑞) = Σ𝑥𝜖𝑋𝑝(𝑥)𝑙𝑜𝑔 (𝑝(𝑥) / 𝑞(𝑥))
KL 散度是Solomon Kullback 和Richard A. Leibler 在1951 年提出的用于衡量2 个分布之间
距离的指标，𝑝 = 𝑞时，𝐷𝐾𝐿 (𝑝|𝑞)取得最小值0。需要注意的是，交叉熵和KL 散度都不是
对称的：
          𝐻(𝑝, 𝑞) ≠ 𝐻(𝑞, 𝑝)
          𝐷𝐾𝐿(𝑝|𝑞) ≠ 𝐷𝐾𝐿 (𝑞|𝑝)
交叉熵可以很好地衡量2 个分布之间的差别，特别地，当分类问题中y 的编码分布𝑝采用
one-hot 编码时：𝐻(𝒚) = 0，此时
          𝐻(𝒚, 𝒐) = 𝐻(𝒚) + 𝐷𝐾𝐿 (𝒚|𝒐) = 𝐷𝐾𝐿(𝒚|𝒐)
退化到真实标签分布y 与输出概率分布o 之间的KL 散度上
根据 KL 散度的定义，我们推导分类问题中交叉熵的计算表达式：
          𝐻(𝒚, 𝒐) = 𝐷𝐾𝐿 (𝒚|𝒐) = Σ𝑗𝑦𝑗 𝑙𝑜𝑔 (𝑦𝑗 / 𝑜𝑗)
         = 1 ∗ 𝑙𝑜𝑔(1 / 𝑜𝑖) + Σ𝑗≠𝑖(0 ∗ 𝑙𝑜𝑔 (0 / 𝑜𝑗))
         = −𝑙𝑜𝑔𝑜𝑖
其中𝑖为One-hot 编码中为1 的索引号，也是当前输入的真实类别。可以看到，ℒ只与真实
类别𝑖上的概率𝑜𝑖有关，对应概率𝑜𝑖越大，𝐻(𝒚, 𝒐)越小，当对应概率为1 时，交叉熵𝐻(𝒚, 𝒐)
取得最小值0，此时网络输出𝒐与真实标签𝒚完全一致，神经网络取得最优状态。最小化交
叉熵的过程也是最大化正确类别的预测概率的过程
"""






























