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

# H函数
# f(𝑥, 𝑦) = (𝑥^2 + 𝑦 − 11)^2 + (𝑥 + 𝑦^2 − 7)^2
def himmelblau(x):
    # himmelblau 函数实现
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

# 通过np.meshgrid 函数(TensorFlow 中也有meshgrid 函数)生成二维平面网格点坐标：
x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
# 生成x-y 平面采样网格点，方便可视化
X, Y = np.meshgrid(x, y)
Z = himmelblau([X, Y]) # 计算网格点上的函数值

# 可视化
fig = plt.figure('himmelblau')
ax = fig.add_subplot(111, projection='3d')
# 或者
# ax = Axe3D(fig)
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# 初始化参数
x = tf.constant([4., 0.])

for step in range(200):# 循环优化200 次
    with tf.GradientTape() as tape: #梯度跟踪
        tape.watch([x]) # 加入梯度跟踪列表
        y = himmelblau(x) # 前向传播
# 反向传播
    grads = tape.gradient(y, [x])[0]
# 更新参数,0.01 为学习率
    x -= 0.01*grads
# 打印优化的极小值
    if step % 20 == 19:
       print ('step {}: x = {}, f(x) = {}'.format(step, x.numpy(), y.numpy()))
































