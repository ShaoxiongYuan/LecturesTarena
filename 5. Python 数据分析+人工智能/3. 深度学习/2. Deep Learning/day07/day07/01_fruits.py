# 01_fruits.py
# 水果分类案例
"""
数据集：通过爬虫爬取的水果图像，共1036张，包含5个类别：
      苹果(288),香蕉(275),葡萄(216)，橙子(276),梨(251)
任务：
1. 对数据集进行预处理，包括：读取图像路径，划分测试集/训练集
2. 搭建CNN，执行训练，保存
3. 模型加载，预测
"""
###################### 数据预处理部分 ######################
import os

# 定义一组公共变量
name_dict = {"apple": 0, "banana": 1, "grape": 2,
             "orange": 3, "pear": 4}
data_root_path = "data/fruits/"  # 数据集所在目录
test_file_path = data_root_path + "test.txt"  # 测试集文件
train_file_path = data_root_path + "train.txt"  # 训练集文件
# 记录每个类别有哪些图片 key:水果名称 value:图片路径构成的列表
name_data_list = {}


# 将图像放入name_data_list字典
def save_train_test_file(path,  # 图片路径
                         name):  # 图片类别
    if name not in name_data_list:  # 该类别水果不在字典中
        img_list = []  # 创建一个空列表
        img_list.append(path)  # 存入第一张图像
        name_data_list[name] = img_list  # 将图像列表存入字典
    else:
        name_data_list[name].append(path)  # 直接添加到列表中


# 遍历数据集下面的每个子目录，将图片存入字典
dirs = os.listdir(data_root_path)
for d in dirs:
    full_path = data_root_path + d  # 拼完整路径

    if os.path.isdir(full_path):  # 如果是子目录
        imgs = os.listdir(full_path)  # 列出子目录下所有的文件
        for img in imgs:
            # 拼接完整路径，并调用函数填入临时字典
            save_train_test_file(full_path + "/" + img,  # 图片路径
                                 d)  # 目录名称(同时也是类别)
    else:  # 如果是文件
        pass

# 将name_data_list中的内容写入测试集/训练集
with open(test_file_path, "w") as f:  # 清空测试集文件
    pass
with open(train_file_path, "w") as f:  # 清空训练集
    pass

# 遍历name_data_list，写入测试集/训练集
for name, img_list in name_data_list.items():
    i = 0
    num = len(img_list)  # 获取每个类别图像数量
    print("%s: %d张" % (name, num))

    # 遍历图像列表
    for img in img_list:
        if i % 10 == 0:  # 每10笔写一笔测试集
            with open(test_file_path, "a") as f:
                # 拼一行样本，格式： apple/1.jpg   0
                line = "%s\t%d\n" % (img, name_dict[name])
                f.write(line)
        else:  # 其余写入训练集
            with open(train_file_path, "a") as f:
                # 拼一行样本，格式： apple/1.jpg   0
                line = "%s\t%d\n" % (img, name_dict[name])
                f.write(line)
        i += 1  # 计数器加1

print("数据预处理完成.")

###################### 模型搭建、训练、保存 ######################
import paddle
import paddle.fluid as fluid
import numpy
import sys
import os
from multiprocessing import cpu_count
import time
import matplotlib.pyplot as plt


def train_mapper(sample):
    """
    根据传入的文本样本，读取图片, 并对图片设置大小、归一化处理
    :param sample: 文本样本，元组，格式为 (图片路径，类别)
    :return: 返回经过归一化处理的图片数据
    """
    img_path, label = sample  # img为路径, label为类别
    if not os.path.exists(img_path):
        print("图片不存在:", img_path)

    # 读取图片内容
    img = paddle.dataset.image.load_image(img_path)
    # 对图像大小进行变换，设置成固定大小
    img = paddle.dataset.image.simple_transform(
                        im=img,  # 原始图像数据
                        resize_size=100,  # 设置成100*100大小
                        crop_size=100,  # 裁剪图像大小
                        is_color=True,  # 彩色图像
                        is_train=True)  # 训练模型，影响裁剪策略
    # 归一化(避免梯度消失、模型更稳定、防止过拟合)
    img = img.astype("float32") / 255.0

    return img, label  # 返回图像数据、标签


# 定义读取器
def train_r(train_list, buffered_size=1024):
    def reader():  # 内部函数逐行读取样本文件
        with open(train_list, "r") as f:
            # 读取所有行，并去除空格
            lines = [line.strip() for line in f]

            for line in lines:
                line = line.replace("\n", "")  # 去掉换行符
                img_path, lab = line.split("\t")  # 拆分
                yield img_path, int(lab)  # 返回图片路径,类别

    return paddle.reader.xmap_readers(
                                train_mapper,  # 将reader读取的数据进一步处理
                                reader,  # 读取的数据交给train_mapper处理
                                cpu_count(),  # 线程数量
                                buffered_size)  # 缓冲区大小


# 数据准备
BATCH_SIZE = 32  # 批次大小
train_reader = train_r(train_list=train_file_path)  # 原始读取器
random_train_reader = paddle.reader.shuffle(
                            reader=train_reader,
                            buf_size=1300)  # 随机读取器
batch_train_reader = paddle.batch(random_train_reader,
                                  batch_size=BATCH_SIZE)
# 变量
image = fluid.layers.data(name="image",
                          shape=[3, 100, 100],
                          dtype="float32")
label = fluid.layers.data(name="label",
                          shape=[1],
                          dtype="int64")


# 搭建CNN
# 结构： 输入层 --> 卷积/激活/池化/dropout -->
#       卷积/激活/池化/dropout --> 卷积/激活/池化/dropout -->
#       fc --> dropout --> fc(softmax)
def convolution_neural_network(image,  # 输入图像数据
                               type_size):  # 输出值的个数
    # 第一组卷积池化
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=image,  # 输入
        filter_size=3,  # 卷积核大小
        num_filters=32,  # 卷积核数量
        pool_size=2,  # 池化区域大小
        pool_stride=2,  # 池化步长值
        act="relu")  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_1,  # 输入
                                dropout_prob=0.5)  # 丢弃率
    # 第二组卷积池化
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=drop,  # 输入
        filter_size=3,  # 卷积核大小
        num_filters=64,  # 卷积核数量
        pool_size=2,  # 池化区域大小
        pool_stride=2,  # 池化步长值
        act="relu")  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_2,  # 输入
                                dropout_prob=0.5)  # 丢弃率
    # 第三组卷积池化
    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input=drop,  # 输入
        filter_size=3,  # 卷积核大小
        num_filters=64,  # 卷积核数量
        pool_size=2,  # 池化区域大小
        pool_stride=2,  # 池化步长值
        act="relu")  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_3,  # 输入
                                dropout_prob=0.5)  # 丢弃率
    # 全连接层
    fc = fluid.layers.fc(input=drop,  # 输入
                         size=512,  # 输出值的个数
                         act="relu")  # 激活函数
    # dropout
    drop = fluid.layers.dropout(x=fc, dropout_prob=0.5)
    # 输出层(fc)
    predict = fluid.layers.fc(input=drop,
                              size=type_size,  # 输出值个数等于分类数量
                              act="softmax")  # 输出层激活函数采用softmax
    return predict


# 调用函数，创建CNN
predict = convolution_neural_network(image=image,  # 图像
                                     type_size=5)  # 分类数量
# 损失函数：交叉熵
cost = fluid.layers.cross_entropy(input=predict,  # 预测值
                                  label=label)  # 真实值
avg_cost = fluid.layers.mean(cost)  # 求均值
# 准确率
accuracy = fluid.layers.accuracy(input=predict,  # 预测值
                                 label=label)  # 真实值
# 优化器
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)

# 执行器
place = fluid.CUDAPlace(0)  # GPU设备运行
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
# feeder
feeder = fluid.DataFeeder(feed_list=[image, label],  # 需喂入的参数
                          place=place)

costs = []  # 记录训练过程中的损失值
accs = []  # 记录训练过程中的准确率
times = 0
batches = []  # 迭代次数

# 开始训练
for pass_id in range(5):
    train_cost = 0  # 临时变量，记录每次训练的损失值

    # 循环读取样本，执行训练
    # enumerate函数自动对迭代出的每笔数据编号
    for batch_id, data in enumerate(batch_train_reader()):
        times += 1

        train_cost, train_acc = exe.run(
            program=fluid.default_main_program(),
            feed=feeder.feed(data),  # 通过feeder喂入数据
            fetch_list=[avg_cost, accuracy])  # 获取损失值，准确率

        if batch_id % 20 == 0:
            print("pass_id:%d, step:%d, cost:%f, acc:%f"
                  % (pass_id, batch_id,
                     train_cost[0], train_acc[0]))
            accs.append(train_acc[0])  # 记录准确率
            costs.append(train_cost[0])  # 记录损失值
            batches.append(times)  # 记录迭代次数

# 训练结束后，保存模型
model_save_path = "mode/fruits/"  # 模型保存路径
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
fluid.io.save_inference_model(dirname=model_save_path,
                              feeded_var_names=["image"],
                              target_vars=[predict],
                              executor=exe)
print("模型保存成功.")

# 训练过程可视化
plt.figure("training")
plt.title("training", fontsize=24)
plt.xlabel("iter", fontsize=20)
plt.ylabel("cost/acc", fontsize=20)
plt.plot(batches, costs, color='red', label="Training Cost")
plt.plot(batches, accs, color='green', label="Training Acc")
plt.legend()
plt.grid()
plt.savefig("train.png")
plt.show()

###################### 测试 ######################
from PIL import Image

def load_img(path): # 根据指定的路径读取图像数据
    img = paddle.dataset.image.load_and_transform(
                    path, 100, 100, False).astype("float32")
    img = img / 255.0 # 归一化
    return img

# 定义执行器
place = fluid.CPUPlace() # 测试可以不使用GPU
infer_exe = fluid.Executor(place) # 重新定义一个Executor

infer_imgs = [] # 存放要预测的图像
test_img = "apple_1.png" # 待预测的图像
infer_imgs.append(load_img(test_img)) # 读取图像数据并添加到待预测列表
# 将列表转换为数组，因为CNN在执行预测的时候，需要传入一个张量
# 实际传入数据/张量可以的，但是不能传入列表
infer_imgs = numpy.array(infer_imgs)

# 加载模型
infer_program, feed_target_names, fetch_targets = \
    fluid.io.load_inference_model(model_save_path,
                                  infer_exe)
# 执行预测
results = infer_exe.run(
        infer_program, #专门用于预测的program
        feed={feed_target_names[0]:infer_imgs}, # 根据返回值赋值
        fetch_list=fetch_targets) # 根据返回值获取结果
print(results)

# 将预测结果转换为更容易阅读的字符串
result = numpy.argmax(results[0]) # 找出预测结果中最大索引值
for k, v in name_dict.items():
    if result == v: # 如果值相等，则打印名称
        print("预测结果：", k)

# 显示待预测的图片
img = Image.open(test_img)
plt.imshow(img)
plt.show()