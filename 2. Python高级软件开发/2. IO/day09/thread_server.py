"""
thread_server.py 基于多线程的网络模型
重点代码

创建监听套接字
循环接收客户端连接请求
当有新的客户端连接创建线程处理客户端请求
主线程继续等待其他客户端连接
当客户端退出，则对应分支线程退出
"""

from socket import *
from threading import Thread


# 全局变量
HOST = '0.0.0.0'
PORT = 8888
ADDR = (HOST,PORT)

# 处理客户端的具体请求
def handle(c):
    while True:
        data = c.recv(1024).decode()
        if not data:
            break
        print(data)
        c.send(b'OK')
    c.close()


# 创建监听套接字
s = socket()
s.bind(ADDR)
s.listen(3)

print("Listen the port 8888")


# 循环等待客户端链接
while True:
    c,addr = s.accept()
    print("Connect from",addr)

    # 创建线程处理客户端请求
    t = Thread(target=handle,args=(c,))
    t.setDaemon(True) # 分支线程随主线程退出
    t.start()

s.close()
