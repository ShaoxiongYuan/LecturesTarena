# 04_mnist.py
# 使用神经网络实现手写体识别
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pylab

# 1.读取数据, 定义变量
mnist = input_data.read_data_sets("MNIST_data/", # 数据集目录
                                  one_hot=True) # 标签是否采用独热编码
x = tf.placeholder(tf.float32, [None, 784]) # N个样本,每个样本784个特征
y = tf.placeholder(tf.float32, [None, 10]) # 标签, N个样本, 每个样本10个概率

w = tf.Variable(tf.random_normal([784, 10])) # 权重
b = tf.Variable(tf.zeros([10])) # 偏置

# 2.构建模型, 损失函数, 定义优化器
pred_y = tf.nn.softmax(tf.matmul(x, w) + b)
# 交叉熵
cross_entropy = -tf.reduce_sum(y * tf.log(pred_y),
                               reduction_indices=1)
# 求均值
cost = tf.reduce_mean(cross_entropy)
# 优化器
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# 3.执行训练, 模型评估
training_epochs = 80 # 训练轮次
batch_size = 100 # 每个批次样本数量
saver = tf.train.Saver()
model_path = "../model/mnist/mnist_model.ckpt" # 模型路径

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 初始化

    # 循环训练
    for epoch in range(training_epochs):
        avg_cost = 0.0
        # 计算每轮下训练的总批次
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            # 获取一个批次的样本
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            params = {x:batch_xs, y:batch_ys} # 训练传入的参数

            o, c = sess.run([optimizer, cost],
                            feed_dict=params) # 训练

            avg_cost += (c / total_batch) # 求平均损失值

        print("epoch: %d, cost=%.9f" % (epoch, avg_cost))
    print("训练结束.")

    # 模型评估
    correct_pred = tf.equal(tf.argmax(pred_y, 1),
                            tf.argmax(y, 1)) # 比较预测结果和真实结果
    # 计算准确率(所有值相加除以元素个数,正好等于均值)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print("accuracy:", accuracy.eval({x:mnist.test.images,
                                      y:mnist.test.labels}))
    # 保存模型
    save_path = saver.save(sess, model_path)
    print("Model Saved:", save_path)

# 4.执行预测
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path) # 加载模型

    # 从测试集中读取两个样本,执行预测
    batch_xs, batch_ys = mnist.test.next_batch(2)
    # 定义预测操作并执行
    output = tf.argmax(pred_y, 1) # 预测, 并取出最大概率索引作为最终结果
    output_val, predv = sess.run([output, pred_y],
                                 feed_dict={x:batch_xs})
    print("预测结论:\n", output_val, "\n")
    print("实际结果:\n", batch_ys, "\n")
    print("预测概率:\n", predv, "\n")

    # 显示参与测试的图像
    im = batch_xs[0]  # 第1个测试样本数据
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]  # 第2个测试样本数据
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
