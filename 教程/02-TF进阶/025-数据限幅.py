import  matplotlib
from 	matplotlib import pyplot as plt
import  tensorflow as tf
from   keras import datasets
import  os
from tensorflow import keras

#考虑怎么实现非线性激活函数ReLU 的问题。它其实可以通过简单的数据限幅运算实现，限制数据的范围𝑥 ∈ [0, +∞)即可
x = tf.range(9)
tf.maximum(x,2) # 下限幅2，将小于2的数据替换为2
tf.minimum(x,7) # 上限幅7，将大于7的数据替换为7
tf.minimum(tf.maximum(x,2),7) # 限幅为2~7
#relu函数
def relu(x):
  return tf.minimum(x,0.) # 下限幅为0 即可
#也可以这样限幅
tf.clip_by_value(x,2,7) # 限幅为2~7























