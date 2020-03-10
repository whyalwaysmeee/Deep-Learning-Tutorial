import  matplotlib
from 	matplotlib import pyplot as plt
import  tensorflow as tf
from   keras import datasets
import  os
from tensorflow import keras

#填充
#为了方便计算维度长度的不同的张量，需要把张量扩张为相同长度，复制操作会破坏原数据结构
"""
填充操作可以通过tf.pad(x, paddings)函数实现，paddings 是包含了多个
[𝐿𝑒𝑓𝑡 𝑃𝑎𝑑𝑑𝑖𝑛𝑔, 𝑅𝑖𝑔ℎ𝑡 𝑃𝑎𝑑𝑑𝑖𝑛𝑔]的嵌套方案List，如[[0,0], [2,1], [1,2]]表示第一个维度不填
充，第二个维度左边(起始处)填充两个单元，右边(结束处)填充一个单元，第三个维度左边
填充一个单元，右边填充两个单元。
"""
a = tf.constant([1,2,3,4,5,6])
b = tf.constant([7,8,1,6])
#在第二个句子的第一个维度的右边填充2个单元
b = tf.pad(b, [[0,2]])

"""
在自然语言处理中，需要加载不同句子长度的数据集，有些句子长度较小，如10 个单
词左右，部份句子长度较长，如超过100 个单词。为了能够保存在同一张量中，一般会选
取能够覆盖大部分句子长度的阈值，如80 个单词：对于小于80 个单词的句子，在末尾填
充相应数量的0；对大于80 个单词的句子，截断超过规定长度的部分单词。以IMDB 数据
集的加载为例
"""
total_words = 10000 # 设定词汇量大小
max_review_len = 80 # 最大句子长度
embedding_len = 100 # 词向量长度
# 加载IMDB数据集
(x_train, y_train), (x_test, y_test) =keras.datasets.imdb.load_data(num_words=total_words)
# 将句子填充或截断到相同长度，设置为末尾填充和末尾截断方式
x_train = keras.preprocessing.sequence.pad_sequences(x_train,
maxlen=max_review_len,truncating='post',padding='post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test,
maxlen=max_review_len,truncating='post',padding='post')

"""
考虑对图片的高宽维度进行填充。以28x28 大
小的图片数据为例，如果网络层所接受的数据高宽为32x32，则必须将28x28 大小填充到
32x32，可以在上、下、左、右方向各填充2 个单元，
"""
x = tf.random.normal([4,28,28,1])
#上述填充方案可以表达为[[0,0], [2,2], [2,2], [0,0]]
tf.pad(x,[[0,0],[2,2],[2,2],[0,0]])

#复制
"""
通过 tf.tile 函数可以在任意维度将数据重复复制多份，如shape 为[4,32,32,3]的数据，
复制方案multiples=[2,3,3,1]，即通道数据不复制，高宽方向分别复制2 份，图片数再复制
1 份:
"""
x = tf.random.normal([4,32,32,3])
tf.tile(x,[2,3,3,1]) # 数据复制
#结果的shape=(8, 96, 96, 3)









