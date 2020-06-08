# 01_saver_demo.py
# 利用saver实现线性模型的增量训练
# 12_linear_regression.py
# 线性回归综合案例
import tensorflow as tf
import os

# 第一步: 创建样本数据
x = tf.random_normal([100, 1], mean=1.75, stddev=0.5,
                     name="x_data")
y_true = tf.matmul(x, [[2.0]]) + 5.0 # 计算 y=2x+5

# 第二步: 定义线性模型, 执行预测 y = wx + b
## 初始化w和b
weight = tf.Variable(tf.random_normal([1,1], name="w"),
                     trainable=True) # 训练过程中是否允许变化
bias = tf.Variable(0.0, name="b", trainable=True)
y_predict = tf.matmul(x, weight) + bias # 计算预测结果

# 第三步: 构建损失函数
loss = tf.reduce_mean(tf.square(y_true - y_predict))#均方差
# 定义优化器(执行梯度下降), 将损失函数的值优化到最小
# 在梯度下降过程中, 会去调整w,b两个参数
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 第四步: 执行训练(使用梯度下降优化器, 反复优化损失函数,
#               将损失函数值优化到最小, 这样就找到了最优参数)
tf.summary.scalar("losses", loss) # 收集损失函数的值

saver = tf.train.Saver() # 示例化Saver对象

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 创建文件写入器,并制定事件文件路径
    fw = tf.summary.FileWriter("../summary/", graph=sess.graph)

    # 训练之前, 检查模型文件是否存在, 如果存在则加载
    if os.path.exists("../model/linear_model/checkpoint"):
        saver.restore(sess, "../model/linear_model/")

    for i in range(200):
        sess.run(train_op)
        summary = sess.run(tf.summary.merge_all()) #摘要合并
        fw.add_summary(summary, i) # 将第i次收集的数据写入事件文件
        print(i, ":", " w:", weight.eval(), ", b:", bias.eval())

    # 训练结束之后,保存模型
    saver.save(sess, "../model/linear_model/")
