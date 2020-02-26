"""
使用udp套接字完成
客户端循环输入单词，获取单词解释
服务端负责接收单词，查询，将单词解释内容给客户端

"""

from socket import *

# 查询单词
def find_word(word):
    # 打开文件
    f = open("dict.txt")  # 默认就是r方式

    # 每次获取一行
    for line in f:
        w = line.split(' ')[0]  # 提取每一行的单词
        # 提高一点效率 如果遍历到的单词已经比目标大就没有必要继续向下找了
        if w > word:
            f.close()
            return "没有找到该单词"
        elif w == word:
            f.close()
            return line
    else:
        f.close()
        return "没有找到该单词"


# 创建套接字
sockfd = socket(AF_INET,SOCK_DGRAM)

#绑定地址
server_addr = ('127.0.0.1',8888)
sockfd.bind(server_addr)

# 循环接收单词
while True:
    data,addr = sockfd.recvfrom(1024) # data --> word
    result = find_word(data.decode()) # 字节串--》字符串
    sockfd.sendto(result.encode(),addr)

sockfd.close()