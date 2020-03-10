from __future__ import absolute_import, division, print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import  matplotlib
from keras import datasets


#用于可视化的参数
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus']=False

#预处理函数
"""
从keras.datasets 中经.batch()后加载的图片x shape 为
[𝑏, 28,28]，像素使用0~255 的整形表示；标注shape 为[𝑏]，即采样的数字编码方式。实际
的神经网络输入，一般需要将图片数据标准化到[0,1]或[−1,1]等0 附近区间，同时根据网
络的设置，需要将shape [28,28] 的输入Reshape 为合法的格式；对于标注信息，可以选择
在预处理时进行one-hot 编码，也可以在计算误差时进行one-hot 编码。
根据下一节的实战设定，我们将MNIST 图片数据映射到𝑥 ∈ [0,1]区间，视图调整为
[𝑏, 28 ∗ 28]；对于标注y，我们选择在预处理函数里面进行one-hot 编码
"""
def preprocess(x, y):
    # [b, 28, 28], [b]
    print(x.shape, y.shape)
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)

    return x, y

#导入手写数字图片数据集，用于分类任务
"""
通过load_data()会返回相应格式的数据，对于图片数据集MNIST, CIFAR10 等，会返回2
个tuple，第一个tuple 保存了用于训练的数据x,y 训练集对象；第2 个tuple 则保存了用于
测试的数据x_test,y_test 测试集对象，所有的数据都用Numpy.array 容器承载
"""
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)

#转换成dataset对象
train_db = tf.data.Dataset.from_tensor_slices((x, y))

#随机打散
"""
通过 Dataset.shuffle(buffer_size)工具可以设置Dataset 对象随机打散数据之间的顺序，
防止每次训练时数据按固定顺序产生，从而使得模型尝试“记忆”住标签信息
buffer_size 指定缓冲池的大小，一般设置为一个较大的参数即可
"""
train_db = train_db.shuffle(1000)

#批训练。一般在网络的计算过程中会同时计算多个样本，我们把这种训练方式叫做批训练，其中样本的数量叫做batchsz
batchsz = 512
train_db = train_db.batch(batchsz)

#直接map预处理函数完成预处理
train_db = train_db.map(preprocess)

#循环训练
"""
一般把完成一个batch 的数据训练，叫做一个step；
测试版(20191108)
5.8 MNIST 测试实战[在此处键入] 25
通过多个step 来完成整个训练集的一次迭代，叫做一个epoch。在实际训练时，通常需要
对数据集迭代多个epoch 才能取得较好地训练效果：

for epoch in range(20): # 训练Epoch 数
   for step, (x,y) in enumerate(train_db): # 迭代Step 数
       a = a + 1 # training...
"""
#也可以通过设置
train_db = train_db.repeat(20)

#处理测试集
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(batchsz).map(preprocess)

#next返回迭代器的下一个项目
x, y = next(iter(train_db))
print('train sample:', x.shape, y.shape)

def main():
    # learning rate
    lr = 1e-2
    accs, losses = [], []

    # 784 => 512
    w1, b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 512 => 256
    w2, b2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1)), tf.Variable(tf.zeros([128]))
    # 256 => 10
    w3, b3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

    for step, (x, y) in enumerate(train_db):

        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 784))

        with tf.GradientTape() as tape:

            # layer1.
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            # layer2
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # output
            out = h2 @ w3 + b3
            # out = tf.nn.relu(out)

            # compute loss
            # [b, 10] - [b, 10]
            loss = tf.square(y - out)
            # [b, 10] => scalar
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        for p, g in zip([w1, b1, w2, b2, w3, b3], grads):
            p.assign_sub(lr * g)

        # print
        if step % 80 == 0:
            print(step, 'loss:', float(loss))
            losses.append(float(loss))

        if step % 80 == 0:
            # evaluate/test
            total, total_correct = 0., 0

            for x, y in test_db:
                # layer1.
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)
                # layer2
                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # output
                out = h2 @ w3 + b3
                # [b, 10] => [b]
                pred = tf.argmax(out, axis=1)
                # convert one_hot y to number y
                y = tf.argmax(y, axis=1)
                # bool type
                correct = tf.equal(pred, y)
                # bool tensor => int tensor => numpy
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]

            print(step, 'Evaluate Acc:', total_correct / total)

            accs.append(total_correct / total)

    plt.figure()
    x = [i * 80 for i in range(len(losses))]
    plt.plot(x, losses, color='C0', marker='s', label='训练')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('train.svg')

    plt.figure()
    plt.plot(x, accs, color='C1', marker='s', label='测试')
    plt.ylabel('准确率')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('test.svg')


if __name__ == '__main__':
    main()
















