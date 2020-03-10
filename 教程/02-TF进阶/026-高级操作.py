import  matplotlib
from 	matplotlib import pyplot as plt
import  tensorflow as tf
from   keras import datasets
import  os
from tensorflow import keras

#tf.gather可以实现根据索引号收集数据的目的
#考虑班级成绩册的例子，共有4 个班级，每个班级35 个学生，8 门科目，保存成绩册的张量shape 为[4,35,8]
x = tf.random.uniform([4,35,8],maxval=100,dtype=tf.int32)
#现在需要收集第1-2个班级的成绩册，可以直接切片实现
x[:2]
#但是对于不规则的索引方式，比如，需要抽查所有班级的第1,4,9,12,13,27 号同学的成绩，则切片方式实现起来非常麻烦，而tf.gather 则是针对于此需求设计的，使用起来非常方便：
tf.gather(x,[0,3,8,11,12,26],axis=1)
#如果需要收集所有同学的第3，5 等科目的成绩，则可以
tf.gather(x,[2,4],axis=2)
#如果希望抽查第[2,3]班级的第[3,4,6,27]号同学的科目成绩，则可以通过组合多个tf.gather 实现。首先抽出第[2,3]班级：
students=tf.gather(x,[1,2],axis=0)
#再从这2 个班级的同学中提取对应学生成绩：
tf.gather(students,[2,3,5,26],axis=1)
#如果希望抽查第2 个班级的第2 个同学的所有科目，第3 个班级的第3 个同学的所有科目，第4 个班级的第4 个同学的所有科目
#方法1：逐个手动提取然后合并
tf.stack([x[1,1],x[2,2],x[3,3]],axis=0)

#tf.gather_nd
#方法2：利用tf.gather_nd，可以通过指定每次采样的坐标来实现采样多个点的目的
tf.gather_nd(x,[[1,1],[2,2],[3,3]])
#一般地，在使用tf.gather_nd 采样多个样本时，如果希望采样第i 号班级，第j 个学生，第k 门科目的成绩
tf.gather_nd(x,[[1,1,2],[2,2,3],[3,3,4]])

#tf.boolean_mask
#除了可以通过给定索引号的方式采样，还可以通过给定掩码(mask)的方式采样
x = tf.random.uniform([4,35,8],maxval=100,dtype=tf.int32)
#采样第1 和第4 个班级，通过tf.boolean_mask(x, mask, axis)可以在axis 轴上根据mask 方案进行采样
#注意掩码的长度必须与对应维度的长度一致
tf.boolean_mask(x,mask=[True, False,False,True],axis=0)
#如果对 8 门科目进行掩码采样
tf.boolean_mask(x,mask=[True,False,False,True,True,False,False,True],axis=2)
"""
考虑与tf.gather_nd 类似方式的多维掩码采样方式。为了方便演示，我们将
班级数量减少到2 个，学生的数量减少到3 个，即一个班级只有3 个学生，shape 为
[2,3,8]。如果希望采样第1 个班级的第1-2 号学生，第2 个班级的第2-3 号学生
"""
x = tf.random.uniform([2,3,8],maxval=100,dtype=tf.int32)
tf.gather_nd(x,[[0,0],[0,1],[1,1],[1,2]]) # 多维坐标采集
#如果使用掩码
tf.boolean_mask(x,[[True,True,False],[False,True,True]])

#tf.where
#例子1：
"""
通过 tf.where(cond, a, b)操作可以根据cond 条件的真假从a 或b 中读取数据，条件判定
规则如下：
若𝑐𝑜𝑛𝑑𝑖为𝑇𝑟𝑢𝑒，𝑜𝑖 = 𝑎𝑖 
若𝑐𝑜𝑛𝑑𝑖为𝐹𝑎𝑙𝑠𝑒，𝑜𝑖 = 𝑏𝑖
其中 i 为张量的索引，返回张量大小与a,b 张量一致，当对应位置中𝑐𝑜𝑛𝑑𝑖为True，𝑜𝑖位置
从𝑎𝑖中复制数据；当对应位置中𝑐𝑜𝑛𝑑𝑖为False，𝑜𝑖位置从𝑏𝑖中复制数据。考虑从2 个全1、
全0 的3x3 大小的张量a,b 中提取数据，其中cond 为True 的位置从a 中对应位置提取，
cond 为False 的位置从b 对应位置提取：
"""
a = tf.ones([3,3]) # 构造a 为全1
b = tf.zeros([3,3]) # 构造b 为全0
# 构造采样条件
cond = tf.constant([[True,False,False],[False,True,False],[True,True,False]])
tf.where(cond,a,b) # 根据条件从a,b 中采样
#例子2：
cond = tf.constant([[True,False,False],[False,True,False],[True,True,False]])
#其中True 共出现4 次，每个True 位置处的索引分布为[0,0], [1,1], [2,0], [2,1]，可以直接通过tf.where(cond)来获得这些索引坐标
tf.where(cond)
#用途：
#我们需要提取张量中所有正数的数据和索引。首先构造张量a
x = tf.random.normal([3,3])
#通过比较运算，得到正数的掩码
mask=x>0 # 比较操作，等同于tf.equal()
#通过tf.where 提取此掩码处True 元素的索引
indices=tf.where(mask) # 提取所有大于0 的元素索引
#拿到索引后，通过tf.gather_nd 即可恢复出所有正数的元素
tf.gather_nd(x,indices) # 提取正数的元素值
#当我们得到掩码mask 之后，也可以直接通过tf.boolean_mask 获取对应元素
tf.boolean_mask(x,mask) # 通过掩码提取正数的元素值

#scatter_nd








