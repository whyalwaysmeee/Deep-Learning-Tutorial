import  matplotlib
from 	matplotlib import pyplot as plt
import  tensorflow as tf
from   keras import datasets
import  os


"""
为了计算分类任务的准确率等指标，一般需要将预测结果和真实标签比较，统计比较
结果中正确的数量来就是计算准确率
"""
out = tf.random.normal([100,10])
out = tf.nn.softmax(out, axis=1) # 输出转换为概率
pred = tf.argmax(out, axis=1) # 选取预测值
y = tf.random.uniform([100],dtype=tf.int64,maxval=10) #真实值
out = tf.equal(pred,y) #比较张量是否相等
out = tf.cast(out,dtype = tf.float32)  #将布尔值转化为整型便于统计
correct = tf.reduce_sum(out)   #统计张量中1的个数
"""
常用比较函数
函数                             功能
tf.math.greater                 𝑎 > 𝑏
tf.math.less                    𝑎 < 𝑏
tf.math.greater_equal           𝑎 ≥ 𝑏
tf.math.less_equal              𝑎 ≤ 𝑏
tf.math.not_equal               𝑎 ≠ 𝑏
tf.math.is_nan                  𝑎 = 𝑛𝑎𝑛
"""