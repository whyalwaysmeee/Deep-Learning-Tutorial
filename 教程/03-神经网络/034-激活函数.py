import  matplotlib
from 	matplotlib import pyplot as plt
import  tensorflow as tf
from   keras import datasets
import  os
from tensorflow import keras
from keras import layers

#Sigmoid
x = tf.linspace(-6.,6.,10)
#tf.nn.sigmoid 实现Sigmoid 函数,向量的范围由[−6,6]映射到[0,1]的区间
y = tf.nn.sigmoid(x)

#ReLU
#tf.nn.relu 实现ReLU 函数,经过ReLU 激活函数后，负数全部抑制为0，正数得以保留
y = tf.nn.relu(x)

#LeakyReLU
#𝐿𝑒𝑎𝑘𝑦𝑅𝑒𝐿𝑈 =  𝑥,    𝑥 ≥ 0
#            𝑝 ∗ 𝑥,     𝑥 < 0
#当𝑝 = 0时，LeayReLU 函数退化为ReLU 函数；当𝑝 ≠ 0时，𝑥 < 0能够获得较小的梯度值𝑝，从而避免出现梯度弥散现象
#tf.nn.leaky_relu 实现LeakyReLU 函数,alpha即p
y = tf.nn.leaky_relu(x, alpha=0.1)

#Tanh
#𝑡𝑎𝑛ℎ(𝑥) =(𝑒^𝑥 − 𝑒^−𝑥) / (𝑒𝑥^ + 𝑒^−𝑥) = 2 ∗ 𝑠𝑖𝑔𝑚𝑜𝑖𝑑(2𝑥) − 1
#tf.nn.tanh 实现tanh 函数,向量的范围被映射到[−1,1]之间
y = tf.nn.tanh(x)



























