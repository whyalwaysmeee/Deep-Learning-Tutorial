import  matplotlib
from 	matplotlib import pyplot as plt
# Default parameters for plots
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus']=False

import  tensorflow as tf
from   keras import datasets
import  os

#核心公式：o𝑢𝑡 = 𝑟𝑒𝑙𝑢{𝑟𝑒𝑙𝑢{𝑟𝑒𝑙𝑢[𝑋@𝑊1 + 𝑏1]@𝑊2 + 𝑏2}@𝑊 + 𝑏3 }

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# x: [60k, 28, 28],
# y: [60k]
(x, y), _ = datasets.mnist.load_data()
# x: [0~255] => [0~1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))


train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)


# [b, 784] => [b, 256] => [b, 128] => [b, 10]
# [dim_in, dim_out], [dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3

losses = []

for epoch in range(20): # iterate db for 10
    for step, (x, y) in enumerate(train_db): # for every batch
        # x:[128, 28, 28]
        # y: [128]

        # 首先将shape 为[𝑏, 28,28]的输入数据Reshape 为[𝑏, 784]，即[b, 28, 28] => [b, 28*28]
        x = tf.reshape(x, [-1, 28*28])

        with tf.GradientTape() as tape: # tf.Variable
            # x: [b, 28*28]
            # h1 = x@w1 + b1
            # [b, 784]@[784, 256] + [256] => [b, 256] + [256] => [b, 256] + [b, 256]
            # 完成第一个非线性函数的计算，我们这里显示地进行Broadcasting
            h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
            h1 = tf.nn.relu(h1)
            # [b, 256] => [b, 128]
            # 同样的方法完成第二个和第三个非线性函数的前向计算，输出层可以不使用ReLU 激活函数
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            # [b, 128] => [b, 10]
            out = h2@w3 + b3

            # compute loss
            # out: [b, 10]
            # y: [b] => [b, 10]
            # 将真实的标注张量y 转变为one-hot 编码，并计算与out 的均方差
            y_onehot = tf.one_hot(y, depth=10)
            # mse = mean(sum(y-out)^2)
            # [b, 10]
            loss = tf.square(y_onehot - out)
            # mean: scalar
            loss = tf.reduce_mean(loss)

        # compute gradients
        """
        上述的前向计算过程都需要包裹在with tf.GradientTape() as tape 上下文中，使得前向计算时
        能够保存计算图信息，方便反向求导运算。通过 tape.gradient()函数求得网络参数到梯度信息：
        """
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # print(grads)
        # w1 = w1 - lr * w1_grad
        # 按照𝜃′ = 𝜃 − 𝜂 ∗ 𝜕ℒ / 𝜕𝜃 来更新网络参数
        # assign_sub()将原地(In-place)减去给定的参数值，实现参数的自我更新操作
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])


        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))

    losses.append(float(loss))

plt.figure()
plt.plot(losses, color='C0', marker='s', label='训练')
plt.xlabel('Epoch')
plt.legend()
plt.ylabel('MSE')
plt.savefig('forward.svg')
# plt.show()


