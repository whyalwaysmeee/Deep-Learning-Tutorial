import  matplotlib
from 	matplotlib import pyplot as plt
import  tensorflow as tf
from   keras import datasets
import  os
from tensorflow import keras
from keras import layers

"""
常见输出类型：
𝒐 ∈ 𝑅𝑑 输出属于整个实数空间，或者某段普通的实数空间，比如函数值趋势的预测，
年龄的预测问题等；
𝒐 ∈ [0,1] 输出值特别地落在[0, 1]的区间，如图片生成，图片像素值一般用[0, 1]表
示；或者二分类问题的概率，如硬币正反面的概率预测问题；
𝒐 ∈ [0, 1], 𝑖 𝑜𝑖 = 1 输出值落在[0, 1]的区间，并且所有输出值之和为 1，常见的如多
分类问题，如MNIST 手写数字图片识别，图片属于10 个类别的概率之和为1；
𝒐 ∈ [−1, 1] 输出值在[-1, 1]之间
"""

#普通实数空间
"""
这一类问题比较普遍，像正弦函数曲线预测、年龄的预测、股票走势的预测等都属于
整个或者部分连续的实数空间，输出层可以不加激活函数。误差的计算直接基于最后一层
的输出𝒐和真实值y 进行计算，如采用均方差误差函数度量输出值𝒐与真实值𝒚之间的距
离：
ℒ = 𝑔(𝒐, 𝒚)
其中𝑔代表了某个具体的误差计算函数
"""

#[0,1]区间
"""
输出值属于[0, 1]区间也比较常见，比如图片的生成，二分类问题等。在机器学习中，
一般会将图片的像素值归一化到[0,1]区间，如果直接使用输出层的值，像素的值范围会分
布在整个实数空间。为了让像素的值范围映射到[0,1]的有效实数空间，需要在输出层后添
加某个合适的激活函数𝜎，其中Sigmoid 函数刚好具有此功能
"""

#[0,1]区间，和为1
"""
输出值𝑜𝑖 ∈ [0,1]，所有输出值之和为1，这种设定以多分类问题最为常见.可以通过在输
出层添加Softmax 函数实现。Softmax 函数不仅可以将输出值映射到[0,1]区间，还满
足所有的输出值之和为1 的特性
"""
z = tf.constant([2.,1.,0.1])
tf.nn.softmax(z)
"""
与Dense 层类似，Softmax 函数也可以作为网络层类使用，通过类layers.Softmax(axis=-1)
可以方便添加Softmax 层，其中axis 参数指定需要进行计算的维度。
在 Softmax 函数的数值计算过程中，容易因输入值偏大发生数值溢出现象；在计算交
叉熵时，也会出现数值溢出的问题。为了数值计算的稳定性，TensorFlow 中提供了一个统
一的接口，将Softmax 与交叉熵损失函数同时实现，同时也处理了数值不稳定的异常，一
般推荐使用，避免单独使用Softmax 函数与交叉熵损失函数。函数式接口为
tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)，
其中y_true 代表了one-hot 编码后的真实标签，y_pred 表示网络的预测值，当from_logits 
设置为True 时，y_pred 表示须为未经过Softmax 函数的变量z；当from_logits 设置为False 
时，y_pred 表示为经过Softmax 函数的输出。
"""
z = tf.random.normal([2,10]) # 构造输出层的输出
y_onehot = tf.constant([1,3]) # 构造真实值
y_onehot = tf.one_hot(y_onehot, depth=10) # one-hot 编码
# 输出层未使用Softmax 函数，故from_logits 设置为True
loss = keras.losses.categorical_crossentropy(y_onehot,z,from_logits=True)
loss = tf.reduce_mean(loss) # 计算平均交叉熵损失
"""
也可以利用losses.CategoricalCrossentropy(from_logits)类方式同时实现Softmax 与交叉熵损
失函数的计算：
"""
# 创建Softmax 与交叉熵计算类，输出层的输出z 未使用softmax
criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
loss = criteon(y_onehot,z) # 计算损失

#[-1,1]
#如果希望输出值的范围分布在[−1, 1]，可以简单地使用tanh 激活函数
x = tf.linspace(-6.,6.,10)
tf.tanh(x) # tanh 激活函数







































































