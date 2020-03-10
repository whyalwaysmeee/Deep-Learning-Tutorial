import  matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import datasets
import os
from tensorflow import keras
from keras import layers
import pandas as pd
from tensorflow import losses
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets
import seaborn as sns


"""
实现一个4 层的全连接网络实现二分类任务，网络输入节点数为2，隐藏层的节点数
设计为：25,50,25，输出层2 个节点，分别表示属于类别1 的概率和类别2 的概率
"""

N_SAMPLES = 2000 # 采样点数
TEST_SIZE = 0.3 # 测试数量比率

# 利用工具函数直接生成数据集
X, y = sklearn.datasets.make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
# 将2000 个点按着7:3 分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=TEST_SIZE, random_state=42)
print(X.shape, y.shape)

# 绘图
def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None,dark=False):
    if (dark):
       plt.style.use('dark_background')
    else:
       sns.set_style("whitegrid")
    plt.figure(figsize=(16,12))
    axes = plt.gca()
    axes.set(xlabel="$x_1$", ylabel="$x_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if(XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha = 1,cmap=cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5],cmap="Greys", vmin=0, vmax=.6)
    # 绘制散点图，根据标签区分颜色
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral,edgecolors='none')
    plt.savefig('dataset.svg')
    plt.close()
# 调用make_plot 函数绘制数据的分布，其中X 为2D 坐标，y 为标签
make_plot(X, y, "Classification Dataset Visualization ")

class Layer:
    # 全连接网络层
    def __init__(self, n_input, n_neurons, activation=None, weights=None,
                 bias=None):
        """
        :param int n_input: 输入节点数
        :param int n_neurons: 输出节点数
        :param str activation: 激活函数类型
        :param weights: 权值张量，默认类内部生成
        :param bias: 偏置，默认类内部生成
        """

    # 通过正态分布初始化网络权值，初始化非常重要，不合适的初始化将导致网络不收敛
        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons) * np.sqrt(1 / n_neurons)
        self.bias = bias if bias is not None else np.random.rand(n_neurons) * 0.1
        self.activation = activation  # 激活函数类型，如’sigmoid’
        self.last_activation = None  # 激活函数的输出值o
        self.error = None  # 用于计算当前层的delta 变量的中间变量
        self.delta = None  # 记录当前层的delta 变量，用于计算梯度

# 前向传播
    def activate(self,x):
        # 前向传播
        r = np.dot(x, self.weights) + self.bias  # X@W+b
        # 通过激活函数，得到全连接层的输出o
        self.last_activation = self._apply_activation(r)
        return self.last_activation

# 不同种类的激活函数
    def _apply_activation(self, r):
    # 计算激活函数的输出
        if self.activation is None:
            return r  # 无激活函数，直接返回
    # ReLU 激活函数
        elif self.activation == 'relu':
            return np.maximum(r, 0)
    # tanh
        elif self.activation == 'tanh':
            return np.tanh(r)
    # sigmoid
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r

    def apply_activation_derivative(self, r):
    # 计算激活函数的导数
    # 无激活函数，导数为1
        if self.activation is None:
            return np.ones_like(r)
    # ReLU 函数的导数实现
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r <= 0] = 0.
            return grad
    # tanh 函数的导数实现
        elif self.activation == 'tanh':
            return 1 - r ** 2
    # Sigmoid 函数的导数实现
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        return r
















