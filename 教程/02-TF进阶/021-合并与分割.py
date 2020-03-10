from __future__ import absolute_import, division, print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import  matplotlib
from keras import datasets

#合并
#合并操作可以在任意的维度上进行，唯一的约束是非合并维度的长度必须一致
#在班级维度上拼接
"""
设张量A 保存了某学校1-4 号班级的成绩册，每个班级35 个学生，共8 门科目，则张量A
的shape 为：[4,35,8]；同样的方式，张量B 保存了剩下的6 个班级的成绩册，shape 为
[6,35,8]
"""
a = tf.random.normal([4,2,3]) # 模拟成绩册A
b = tf.random.normal([6,2,3]) # 模拟成绩册B
c = tf.concat([a,b],axis=0)
#在科目维度拼接
"""
张量A 保存了
所有班级所有学生的前4 门科目成绩，shape 为[10,35,4]，张量B 保存了剩下的4 门科目
成绩，shape 为[10,35,4]
"""
a = tf.random.normal([10,35,4])
b = tf.random.normal([10,35,4])
c = tf.concat([a,b],axis=2) # 在科目维度拼接
#堆叠
#在合并数据时创建一个新的维度
"""
张量A 保存了某个班级的成绩册，shape 为[35,8]，张量B 保存了另一个班级的成绩册，
shape 为[35,8]。合并这2 个班级的数据时，需要创建一个新维度，定义为班级维度，
新维度可以选择放置在任意位置，一般根据大小维度的经验法则，将较大概念的班级维度
放置在学生维度之前，则合并后的张量的新shape 应为[2,35,8]。
"""
a = tf.random.normal([35,8])
b = tf.random.normal([35,8])
#tf.stack 也需要满足张量堆叠合并条件，它需要所有合并的张量shape 完全一致才可合并
tf.stack([a,b],axis=0) # 堆叠合并为2个班级

#分割
x = tf.random.normal([10,35,8])
"""
参数设置：
x：待分割张量
axis：分割的维度索引号
num_or_size_splits：切割方案。当num_or_size_splits 为单个数值时，如10，表示切割
为10 份；当num_or_size_splits 为List 时，每个元素表示每份的长度，如[2,4,2,2]表示
切割为4 份，每份的长度分别为2,4,2,2
"""
#等长切割
result = tf.split(x,axis=0,num_or_size_splits=10)
#自定义长度的切割
result = tf.split(x,axis=0,num_or_size_splits=[4,2,2,2])
"""
特别地，如果希望在某个维度上全部按长度为1 的方式分割，还可以直接使用tf.unstack(x,
axis)。这种方式是tf.split 的一种特殊情况，切割长度固定为1，只需要指定切割维度即
可
"""
x = tf.random.normal([10,35,8])
result = tf.unstack(x,axis=0) # Unstack 为长度为1，结果是班级维度消失

















