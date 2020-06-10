# 01_news_classify.py
# 新闻分类
"""
数据来源：从网站上爬取56821条中文新闻摘要
数据类容：包含10类(国际、文化、娱乐、体育、财经、汽车、教育、科技、房产、证券)
任务：
  1）数据预处理：字典编码，文本编码，建立训练集/测试集
  2）模型搭建，训练，评估
  3）执行预测
"""
############################## 数据预处理 ##############################
import os
from multiprocessing import cpu_count
import numpy as np
import paddle
import paddle.fluid as fluid

# 定义公共变量
data_root = "data/news_classify/"  # 数据集所在目录
data_file = "news_classify_data.txt"  # 原始样本文件名称
test_file = "test_list.txt"  # 测试集文件名称
train_file = "train_list.txt"  # 训练集文件名称
dict_file = "dict_txt.txt"  # 编码后的字典文件

data_file_path = data_root + data_file  # 样本文件完整路径
dict_file_path = data_root + dict_file  # 字典文件完整路径
test_file_path = data_root + test_file  # 测试集文件完整路径
train_file_path = data_root + train_file  # 训练集文件完整路径


# 生成字典：将每个汉字编码成一个数字，并保存到字典文件中
def create_dict():
    dict_set = set()  # 集合，用来去重
    with open(data_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 遍历每行
    for line in lines:
        # 取出标题部分，并去除换行符
        title = line.split("_!_")[-1].replace("\n", "")

        for w in title:  # 取出每个字，添加到集合去重
            dict_set.add(w)

    # 遍历集合，为每个字分配一个编码
    dict_list = []  # 临时列表
    i = 0  # 计数器
    for s in dict_set:
        dict_list.append([s, i])  # 将[文字,编码]子列表添加到列表中
        i += 1

    dict_txt = dict(dict_list)  # 将列表转换为字典

    # 考虑未知字符
    end_dict = {"<unk>": i}  # 未知字符
    dict_txt.update(end_dict)  # 将未知字符编码添加到字典中

    # 将字典保存到文件
    with open(dict_file_path, "w", encoding="utf-8") as f:
        f.write(str(dict_txt))

    print("生成字典完成.")


# 对一个标题进行编码
def line_encoding(title,  # 要参与编码的文本
                  dict_txt,  # 编码使用的字典
                  label):  # 文本所属的类别
    new_line = ""  # 编码结果

    for w in title:
        if w in dict_txt:  # 如果字在字典中，则直接取出编码
            code = str(dict_txt[w])
        else:  # 如果不在字典中，取未知字符编码
            code = str(dict_txt["<unk>"])
        new_line = new_line + code + ","  # 将编码追加到字符后免

    new_line = new_line[:-1]  # 去掉最后一个多余的逗号
    # 拼接成一行新的样本
    new_line = new_line + "\t" + label + "\n"

    return new_line


# 对原始样本进行编码： 逐行取出样本，对每个字使用编码值替代
# 编码后的样本写入训练集/测试集
def create_data_list():
    # 清空测试集/训练集文件
    with open(test_file_path, "w") as f:
        pass
    with open(train_file_path, "w") as f:
        pass

    # 从字典文件中读取字典内容，并且创建一个字典对象
    with open(dict_file_path, "r", encoding="utf-8") as f:
        dict_txt = eval(f.readlines()[0])

    # 读取所有的样本
    with open(data_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    for line in lines:
        words = line.replace("\n", "").split("_!_")
        label = words[1]  # 标签
        title = words[3]  # 标题
        # 执行编码
        new_line = line_encoding(title, dict_txt, label)
        if i % 10 == 0:  # 写测试集
            with open(test_file_path, "a", encoding="utf-8") as f:
                f.write(new_line)
        else:  # 写训练集
            with open(train_file_path, "a", encoding="utf-8") as f:
                f.write(new_line)
        i += 1
    print("生成测试集、训练集结束.")

create_dict()
create_data_list()

###################### 模型搭建、训练 ######################
# 读取字典长度
def get_dict_len(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        line = eval(f.readlines()[0])
    return len(line.keys())

def data_mapper(sample):
    data, label = sample # 将sample中的元素拆分到变量中
    # 将data中的元素转换为整数，并存入列表
    val = [int(w) for w in data.split(",")]

    return val, int(label)

# 读取训练集的reader
def train_reader(train_file_path):
    def reader():
        with open(train_file_path, "r") as f:
            lines = f.readlines() # 读取训练集中所有行
            np.random.shuffle(lines) # 打乱所有样本

            for line in lines:
                data, label = line.split("\t")
                yield data, label
    return paddle.reader.xmap_readers(data_mapper,
                                      reader,
                                      cpu_count(),
                                      1024)
# 读取测试集的reader
def test_reader(test_file_path):
    def reader():
        with open(test_file_path, "r") as f:
            lines = f.readlines() # 读取训练集中所有行

            for line in lines:
                data, label = line.split("\t")
                yield data, label
    return paddle.reader.xmap_readers(data_mapper,
                                      reader,
                                      cpu_count(),
                                      1024)

# 定义网络
def CNN_net(data, dict_dim, class_dim=10,
            emb_dim=128, hid_dim=128, hid_dim2=98):
    """
    定义网络
    :param data: 原始文本数据
    :param dict_dim: 词典大小
    :param class_dim: 分类数量
    :param emb_dim: 词嵌入计算参数
    :param hid_dim: 第一组卷积层卷积核数量
    :param hid_dim2: 第二组卷积层卷积核数量
    :return:
    """
    # 词嵌入层：生成词向量
    emb = fluid.layers.embedding(input=data,
                                 size=[dict_dim, emb_dim])
    # 并列两组卷积/池化层
    conv1 = fluid.nets.sequence_conv_pool(input=emb, # 输入
                                    num_filters=hid_dim,#卷积核数量
                                    filter_size=3,#卷积核大小
                                    act="tanh", # 激活函数
                                    pool_type="sqrt")#池化类型
    conv2 = fluid.nets.sequence_conv_pool(input=emb, # 输入
                                    num_filters=hid_dim2,#卷积核数量
                                    filter_size=4,#卷积核大小
                                    act="tanh", # 激活函数
                                    pool_type="sqrt")#池化类型

    # 输出层
    output = fluid.layers.fc(input=[conv1, conv2],# 输入
                             size=class_dim, # 输出值个数
                             act="softmax") # 激活函数
    return output

# 定义张量
words = fluid.layers.data(name="words",
                          shape=[1],
                          dtype="int64",
                          lod_level=1)
label = fluid.layers.data(name="label",
                          shape=[1],
                          dtype="int64")
dict_dim = get_dict_len(dict_file_path) # 获取字典长度
# 创建模型
model = CNN_net(words, dict_dim)
# 定义损失函数
cost = fluid.layers.cross_entropy(input=model, # 预测值
                                  label=label) # 真实值
avg_cost = fluid.layers.mean(cost)
# 准确率
acc = fluid.layers.accuracy(input=model, # 预测值
                            label=label) # 真实值
# 克隆一个test_program用于模型评估
# for_test = True会少做一些优化
test_program = fluid.default_main_program().clone(for_test=True)
# 定义优化器
optimizer = fluid.optimizer.AdagradOptimizer(0.001)
optimizer.minimize(avg_cost)

# 执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
# reader
tr_reader = train_reader(train_file_path)
batch_train_reader = paddle.batch(tr_reader,
                                  batch_size=128)

ts_reader = test_reader(test_file_path)
batch_test_reader = paddle.batch(reader=ts_reader,
                                 batch_size=128)
# feeder
feeder = fluid.DataFeeder(place=place,
                          feed_list=[words, label])

# 开始训练
for pass_id in range(20):
    # 读取训练集中的数据进行训练
    for batch_id, data in enumerate(batch_train_reader()):
        train_cost, train_acc = exe.run(
                                program=fluid.default_main_program(),
                                feed=feeder.feed(data),
                                fetch_list=[avg_cost, acc])
        # 每100次打印一笔
        if batch_id % 100 == 0:
            print("pass_id:%d, batch_id:%d, cost:%f, acc:%f"
                  % (pass_id, batch_id,
                     train_cost[0], train_acc[0]))

    # 每轮训练完成后，对模型进行评估
    test_costs_list = [] # 记录损失值
    test_accs_list = [] # 记录准确率

    # 读取测试集数据，进行预测评估
    for batch_id, data in enumerate(batch_test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                feed=feeder.feed(data),
                                fetch_list=[avg_cost, acc])
        test_costs_list.append(test_cost[0])
        test_accs_list.append(test_acc[0])

    # 计算平均损失值和准确率
    avg_test_cost = sum(test_costs_list)/len(test_costs_list)
    avg_test_acc = sum(test_accs_list)/len(test_accs_list)
    print("pass_id:%d test_cost:%f, test_acc:%f"
          % (pass_id, avg_test_cost, avg_test_acc))

model_save_dir = "model/news_classify/"  # 模型保存路径
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
fluid.io.save_inference_model(model_save_dir, # 模型保存路径
                              feeded_var_names=[words.name], # 使用模型时需传入的参数
                              target_vars=[model], # 预测结果
                              executor=exe) # 执行器
print("模型保存完成.")

######################### 预测 ########################
# 对要预测文本进行编码
def get_data(sentence):
    with open(dict_file_path, "r", encoding="utf-8") as f:
        dict_txt = eval(f.readlines()[0]) # 生成字典对象

    keys = dict_txt.keys()
    ret = [] # 编码后的结果
    for s in sentence: # 遍历句子，取出每个字进行编码
        if not s in keys: # 字不在字典中
            s = "<unk>"
        ret.append(int(dict_txt[s])) # 取出编码，转换为整型
    return ret

# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

print("加载模型")
infer_program, feeded_var_names, target_var = \
    fluid.io.load_inference_model(dirname=model_save_dir,
                                  executor=exe)
# 定义测试文本，并进行编码
texts = [] # 经过编码后的待预测的文本
data1 = get_data("在获得诺贝尔文学奖7年之后，莫言15日晚间在山西汾阳贾家庄如是说")
data2 = get_data("综合'今日美国'、《世界日报》等当地媒体报道，芝加哥河滨警察局表示")
data3 = get_data("中国队无缘2020年世界杯")
data4 = get_data("中国人民银行今日发布通知，降低准备金率，预计释放4000亿流动性")
data5 = get_data("10月20日,第六届世界互联网大会正式开幕")
data6 = get_data("同一户型，为什么高层比低层要贵那么多？")
data7 = get_data("揭秘A股周涨5%资金动向：追捧2类股，抛售600亿香饽饽")
data8 = get_data("宋慧乔陷入感染危机，前夫宋仲基不戴口罩露面，身处国外神态轻松")
data9 = get_data("此盆栽花很好养，花美似牡丹，三季开花，南北都能养，很值得栽培")#不属于任何一个类别
texts.append(data1)
texts.append(data2)
texts.append(data3)
texts.append(data4)
texts.append(data5)
texts.append(data6)
texts.append(data7)
texts.append(data8)
texts.append(data9)

# 获取每个句子的长度
base_shape = [[len(c) for c in texts]]
# 生成lod_tensor
tensor_words = fluid.create_lod_tensor(texts,
                                       base_shape,
                                       place)
# 执行预测
result = exe.run(program=infer_program, # 预测的program
             feed={feeded_var_names[0]:tensor_words},#参数
             fetch_list=target_var)# 获取结果
# print(result)
names = ["文化","娱乐","体育","财经","房产",
         "汽车","教育","科技","国际","证券"]

for i in range(len(texts)):
    # 取出第0页、第i行排序，取出排序后的最后一个元素
    lab = np.argsort(result)[0][i][-1]
    print("预测结果:%d, 名称:%s, 概率:%f"
          %(lab, names[lab], result[0][i][lab]))