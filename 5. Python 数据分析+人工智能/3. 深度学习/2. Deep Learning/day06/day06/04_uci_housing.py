# 04_uci_housing.py
# 波士顿房价预测案例
import paddle
import paddle.fluid as fluid
import numpy as np
import math
import os
import matplotlib.pyplot as plt

BUF_SIZE = 500  # 缓冲区大小
BATCH_SIZE = 20  # 批次大小

# 第一步: 数据准备
## 读取器
random_reader = paddle.reader.shuffle(
    paddle.dataset.uci_housing.train(),  # 训练集reader
    buf_size=BUF_SIZE)  # 随机reader
train_reader = paddle.batch(random_reader,
                            batch_size=BATCH_SIZE)  # 批量reader
## 打印数据
# for sample_data in train_reader():
#     print(sample_data)

x = fluid.layers.data(name="x", shape=[13], dtype="float32")
y = fluid.layers.data(name="y", shape=[1], dtype="float32")

# 第二步: 模型搭建
## fc模型
y_predict = fluid.layers.fc(input=x,  # 输入
                            size=1,  # 输出值个数
                            act=None)  # 激活函数
## 损失函数
cost = fluid.layers.square_error_cost(input=y_predict,
                                      label=y)
avg_cost = fluid.layers.mean(cost)  # 均方差
## 优化器
optimizer = fluid.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_cost)

# 第三步: 定义执行器, 执行训练
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())  # 初始化
## feeder
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])

iter = 0
iters = []  # 记录迭代次数,用于可视化
train_costs = []  # 训练过程中的损失值
model_save_dir = "model/fit_a_line.model"  # 模型保存路径

for pass_id in range(20):
    train_cost = 0
    i = 0
    for data in train_reader():  # 循环读取一个批次数据
        i += 1

        train_cost = exe.run(
            program=fluid.default_main_program(),
            feed=feeder.feed(data),  # 通过feeder喂入参数
            fetch_list=[avg_cost])  # 获取结果

        if i % 20 == 0:
            print("pass_id:%d, cost:%f"
                  % (pass_id, train_cost[0][0]))
        iter = iter + BATCH_SIZE
        iters.append(iter)  # 记录总的训练次数
        train_costs.append(train_cost[0][0])  # 记录损失值

# 训练结束,保存模型
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)  # 目录不存在,则创建
fluid.io.save_inference_model(model_save_dir,  # 保存的目录
                              ["x"],  # 预测时需要传入的参数
                              [y_predict],  # 取回预测结果
                              exe)  # 模型参数位于哪个exe上
# 训练过程可视化
plt.figure("Training Cost", facecolor="gray")
plt.title("Training Cost", fontsize=24)
plt.xlabel("iter", fontsize=14)
plt.ylabel("cost", fontsize=14)
plt.plot(iters, train_costs, color="red", label="Training Cost")
plt.grid()
# plt.show()
plt.savefig("train.png")

# 第四步: 模型预测(使用测试集数据)
infer_exe = fluid.Executor(place)
infer_result = []  # 预测值列表
ground_truths = []  # 真实值列表
## 加载模型
# infer_program: 预测程序(包含了数据、计算规则)
# feed_target_names: 需要喂入的变量
# fetch_targets: 预测结果保存的变量
[infer_program, feed_target_names, fetch_targets] = \
    fluid.io.load_inference_model(model_save_dir,  # 模型保存目录
                                  infer_exe)  # 加载到哪个执行器
# 读取测试数据
infer_reader = paddle.batch(
    paddle.dataset.uci_housing.test(),
    batch_size=200)  # 读取器
test_data = next(infer_reader())  # 读取一个批次数据
test_x = np.array([data[0] for data in test_data]).astype("float32")
test_y = np.array([data[1] for data in test_data]).astype("float32")

x_name = feed_target_names[0]  # 取出模型预测需传入的参数名称

results = infer_exe.run(
    infer_program,  # 预测执行的program
    feed={x_name: np.array(test_x)},  # 喂入的参数
    fetch_list=fetch_targets)  # 预测结果从哪里获取
# 取出预测值
for val in results[0]:
    infer_result.append(val)
# 取出真实结果
for val in test_y:
    ground_truths.append(val)
# 利用预测结果，真实结果绘制散点图
plt.figure("scatter", facecolor="lightgray")
plt.title("TestResult", fontsize=20)
x = np.arange(1, 30)
y = x  # 用来绘制y=x参照线
plt.plot(x, y)  # 绘制y=x参照线
plt.xlabel("ground truth", fontsize=14)
plt.ylabel("inter result", fontsize=14)
plt.scatter(ground_truths, infer_result,
            color="green", label="Test")
plt.grid()
plt.savefig("predict.png")
plt.show()
