"""
file_read.py
文件读取演示
"""

# 打开文件
f = open('file','r')

# 读写文件
# data = f.read(5)
# print(data)

# 循环读取
# while True:
#     data = f.read(1024)
#     # 当读取到文件结尾会读到空字符串，此时结束循环
#     if not data:
#         break
#     print(data)

# 读取一行内容
# data = f.readline(8)
# print(data)
# data = f.readline(8)
# print(data)

# 读取多行内容
# data = f.readlines(15)
# print(data)

# 迭代方式获取文件内容
for line in f:
    print(line) # 每次获取一行

# 关闭文件 f不存在了
f.close()