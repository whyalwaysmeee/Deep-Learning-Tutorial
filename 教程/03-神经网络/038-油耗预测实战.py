import  matplotlib
from 	matplotlib import pyplot as plt
import  tensorflow as tf
from   keras import datasets
import  os
from tensorflow import keras
from keras import layers
import pandas as pd
from tensorflow import losses

dataset_path = keras.utils.get_file("auto-mpg.data",
"https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
na_values = "?", comment='\t',
sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
#print(dataset.head())
#原始数据中的数据可能含有空字段(缺失值)的数据项，需要清除这些记录项：
#print(dataset.isna().sum()) # 统计空白数据
dataset = dataset.dropna() # 删除空白数据项
#print(dataset.isna().sum()) # 再次统计空白数据。清除后，数据集记录项减为392项
origin = dataset.pop('Origin')
# 根据origin 列来写入新的3 个列
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
print(train_dataset)

# 移动MPG 油耗效能这一列为真实标签Y
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# 查看训练集的输入X 的统计数据
train_stats = train_dataset.describe()

train_stats = train_stats.transpose()

# 标准化数据
def norm(x):
   return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
# 打印出训练集和测试集的大小
#print(normed_train_data.shape,train_labels.shape)
#print(normed_test_data.shape, test_labels.shape)

#利用切分的训练集数据构建数据集对象
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values,
train_labels.values)) # 构建Dataset 对象
train_db = train_db.shuffle(100).batch(32) # 随机打散，批量化

class Network(keras.Model):
# 回归网络
  def __init__(self):
     super(Network, self).__init__()
# 创建3 个全连接层
     self.fc1 = layers.Dense(64, activation='relu')
     self.fc2 = layers.Dense(64, activation='relu')
     self.fc3 = layers.Dense(1)
  def call(self, inputs, training=None, mask=None):
# 依次通过3 个全连接层
     x = self.fc1(inputs)
     x = self.fc2(x)
     x = self.fc3(x)
     return x

model = Network() # 创建网络类实例
# 通过build 函数完成内部张量的创建，其中4 为任意的batch 数量，9 为输入特征长度
model.build(input_shape=(4, 9))
model.summary() # 打印网络信息
optimizer = tf.keras.optimizers.RMSprop(0.001) # 创建优化器，指定学习率

for epoch in range(200): # 200 个Epoch
   for step, (x,y) in enumerate(train_db): # 遍历一次训练集
# 梯度记录器
      with tf.GradientTape() as tape:
          out = model(x) # 通过网络获得输出
          loss = tf.reduce_mean(losses.MSE(y, out)) # 计算MSE
          mae_loss = tf.reduce_mean(losses.MAE(y, out)) # 计算MAE
      if step % 10 == 0:  # 打印训练误差
          print(epoch, step, float(loss))
# 计算梯度，并更新
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))







