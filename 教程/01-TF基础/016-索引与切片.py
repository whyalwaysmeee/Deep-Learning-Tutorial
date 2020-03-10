from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import time
import tempfile
from tensorflow import keras


#索引
x = tf.random.normal([4,32,32,3])
#取第1张图片
x[0]
#取第1张图片的第2行
x[0][1]
#取第1张图片的第2行，第3列的像素
x[0][1][2]
#取第3张图片，第2行，第1列的像素，B通道(第2个通道)颜色强度值
x[2][1][0][1]
#维度较高时，可以这样书写
x[1,9,2]

#切片
x = tf.random.normal([4,32,32,3])
#读取第 2,3 张图片
x[1:3]
#读取第1张图片上的所有行
x[0,::]
"""
切片方式格式小结
切片方式             意义
start:end:step     从 start 开始读取到end(不包含end)，步长为step
start:end          从 start 开始读取到end(不包含end)，步长为1
start:             从 start 开始读取完后续所有元素，步长为1
start::step        从 start 开始读取完后续所有元素，步长为step
:end:step          从 0 开始读取到end(不包含end)，步长为step
:end               从 0 开始读取到end(不包含end)，步长为1
::step             每隔 step-1 个元素采样所有
::                 读取所有元素
:                  读取所有元素
特别地，step 可以为负数，考虑最特殊的一种例子，step = −1时，start: end: −1表示
从start 开始，逆序读取至end 结束(不包含end)，索引号𝑒𝑛𝑑 ≤ 𝑠𝑡𝑎𝑟𝑡
"""
#读取G 通道上的数据时，前面所有维度全部提取
x[:,:,:,1]
"""
...切片方式小结
切片方式            意义
a,⋯,b           a 维度对齐到最左边，b 维度对齐到最右边，中间的维度全部读取，其他维度按a/b 的方式读取
a,⋯             a 维度对齐到最左边，a 维度后的所有维度全部读取，a 维度按a 方式读取。这种情况等同于a 索引/切片方式
⋯，b             b 维度对齐到最右边，b 之前的所有维度全部读取，b 维度按b 方式读取
⋯               读取张量所有数据
"""
#读取第 1-2 张图片的G/B 通道数据
x[0:2,...,1:]
#读取最后2 张图片
x[2:,...]
#读取 R/G 通道数据
x[...,:2]
