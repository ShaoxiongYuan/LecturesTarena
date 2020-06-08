# 03_reader_demo.py
# 样本读取示例
"""
1. 自定义一个原始读取器
2. 将自定义读取器包装成随机读取器
3. 将随机读取器包装成批量读取器
"""
import paddle
# 定义原始读取器, 每次从样本中读取一行并返回
def reader_creator(file_path):
    def reader():
        with open(file_path, "r") as f: # 打开文件
            lines = f.readlines()  # 读取所有行
            for line in lines:
                yield line.replace("\n","") # 生成一个数据对并返回
    return reader
reader = reader_creator("test.txt") # 生成器函数,可迭代对象
shuffle_reader = paddle.reader.shuffle(reader, 10)#随机读取器
batch_reader = paddle.batch(shuffle_reader, 3)#批量化,每批次3笔
# for data in reader():
# for data in shuffle_reader():
for data in batch_reader():
    print(data, end="")


