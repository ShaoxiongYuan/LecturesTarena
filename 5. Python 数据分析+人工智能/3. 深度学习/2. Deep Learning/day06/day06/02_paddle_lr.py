# 02_paddle_lr.py
# 线性回归综合案例
import paddle
import paddle.fluid as fluid
import numpy as np
import matplotlib.pyplot as plt

# 定义x,y
train_data = np.array([[0.5], [0.6], [0.8],
                       [1.1], [1.4]]).astype("float32")
y_true = np.array([[5.0], [5.5], [6.0],
                   [6.8], [6.8]]).astype("float32")
# 定义变量
x = fluid.layers.data(name="x", shape=[1], dtype="float32")
y = fluid.layers.data(name="y", shape=[1], dtype="float32")
# 通过全连接模型执行线性计算
y_predict = fluid.layers.fc(input=x, # 输入
                            size=1, # 输出值的个数
                            act=None) # 回归问题,不使用激活函数
# 定义损失函数
cost = fluid.layers.square_error_cost(input=y_predict,#预测y值
                                      label=y)#真实y值
avg_cost = fluid.layers.mean(cost) # 均方差
# 优化器
optimizer = fluid.optimizer.SGD(learning_rate=0.01) # 随机梯度下降优化器
optimizer.minimize(avg_cost)
# 执行器
exe = fluid.Executor(fluid.CPUPlace()) # 执行器
exe.run(fluid.default_startup_program()) # 初始化

# 开始训练
costs = [] # 记录损失值
iters = [] # 记录迭代次数
params = {"x":train_data, "y":y_true} # 参数

for i in range(40):
    # 执行训练, 未指定执行的program, 执行默认program
    outs = exe.run(feed=params, # 参数
                   fetch_list=[y_predict.name, avg_cost.name])
    iters.append(i) # 记录迭代次数
    costs.append(outs[1][0]) # 损失值
    print("i:%d, cost:%f" % (i,outs[1][0]))

# 损失函数可视化
plt.figure("Trainging")
plt.title("Training Cost", fontsize=24)
plt.xlabel("Iter", fontsize=14)
plt.ylabel("Cost", fontsize=14)
plt.plot(iters, costs, color="red", label="Training Cost")
plt.grid()

# 线性模型可视化
tmp = np.random.rand(10, 1)
tmp = tmp * 2
tmp.sort(axis=0)
x_test = np.array(tmp).astype("float32")
params = {"x": x_test, "y":x_test} # 第二个x_test无实际用途,仅仅为了避免语法错误
y_out = exe.run(feed=params, fetch_list=[y_predict.name])
y_test = y_out[0]

# 线性模型可视化
plt.figure("Inference")
plt.title("Linear Regression", fontsize=24)
plt.plot(x_test, y_test, color="red", label="inference")
plt.scatter(train_data, y_true)

plt.legend()
plt.grid()
plt.show()

