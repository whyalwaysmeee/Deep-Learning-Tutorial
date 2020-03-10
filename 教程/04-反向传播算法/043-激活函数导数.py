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

#sigmoid函数导数
#𝜎(𝑥) = 1 / (1 + 𝑒^−𝑥)，导数为 𝜎(1 − 𝜎)
"""
Sigmoid 函数的导数表达式最终可以表达为激活函数的输出值的简单运算，利
用这一性质，在神经网络的梯度计算中，通过缓存每层的Sigmoid 函数输出值，即可在需
要的时候计算出其导数
"""
def sigmoid(x): # sigmoid 函数
    return 1 / (1 + np.exp(-x))
def derivative(x): # sigmoid 导数的计算
    return sigmoid(x)*(1-sigmoid(x))

#ReLU函数导数
#𝑅𝑒𝐿𝑈(𝑥) ≔ 𝑚𝑎𝑥(0, 𝑥)，导数为 1， 𝑥 ≥ 0
#                           0， 𝑥 < 0
"""
ReLU 函数的导数计算简单，x 大于等于零的时候，导数值恒为1，在反向传播
的时候，它既不会放大梯度，造成梯度爆炸(Gradient exploding)；也不会缩小梯度，造成梯
度弥散(Gradient vanishing)
"""
def derivative(x): # ReLU 函数的导数
  d = np.array(x, copy=True) # 用于保存梯度的张量
  d[x < 0] = 0 # 元素为负的导数为0
  d[x >= 0] = 1 # 元素为正的元素导数为1
  return d

#LeakyReLU函数导数
#𝐿𝑒𝑎𝑘𝑦𝑅𝑒𝐿𝑈 = 𝑥，     𝑥 ≥ 0
#           𝑝 ∗ 𝑥， 𝑥 < 0
#导数为 1， 𝑥 ≥ 0
#      𝑝， 𝑥 < 0
def derivative(x, p):
    dx = np.ones_like(x) # 创建梯度张量
    dx[x < 0] = p # 元素为负的导数为p
    return dx

#Tanh 函数梯度
#𝑡𝑎𝑛ℎ(𝑥) =(𝑒^𝑥 − 𝑒^−𝑥) / (𝑒^𝑥 + 𝑒^−𝑥) = 2 ∗ 𝑠𝑖𝑔𝑚𝑜𝑖𝑑(2𝑥) − 1
#导数为 1 − (𝑒^𝑥 − 𝑒^−𝑥)^2 (𝑒^𝑥 + 𝑒^−𝑥)^2 = 1 − 𝑡𝑎𝑛ℎ^2(𝑥)
def sigmoid(x): # sigmoid 函数实现
   return 1 / (1 + np.exp(-x))
def tanh(x): # tanh 函数实现
   return 2*sigmoid(2*x) - 1
def derivative(x): # tanh 导数实现
   return 1-tanh(x)**2




















