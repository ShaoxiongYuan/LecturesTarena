"""
服务端
逻辑处理
"""

from socket import *
from multiprocessing import Process
import  signal,sys
from dict_db import Database

# 定义地址为全局变量
HOST = '0.0.0.0'
PORT = 8000
ADDR = (HOST,PORT)


# 实例化一个对象帮助处理数据库交互的工作 (链接数据库)
db = Database()

# 注册处理函数
def do_register(connfd,name,passwd):
    # 调用数据处理方法判定可否注册
    if db.register(name,passwd):
        connfd.send(b'OK')
    else:
        connfd.send(b'FAIL')

# 处理登录请求
def do_login(connfd,name,passwd):
    if db.login(name, passwd):
        connfd.send(b'OK')
    else:
        connfd.send(b'FAIL')

# 处理单词查询
def do_query(connfd,name,word):

    db.insert_history(name,word)  # 插入历史记录

    mean = db.query(word)  # 查单词
    msg = "%s : %s"%(word,mean)
    connfd.send(msg.encode())


# 处理客户端请求
def handle(connfd):
    db.create_cursor() # 每个进程创建各自的游标
    while True:
        data = connfd.recv(1024).decode()  # 接收请求
        tmp = data.split(' ')
        if not tmp or tmp[0] == 'E':
            connfd.close()
            db.cur.close()
            return
        elif tmp[0] == 'R':
            # "R name password"
            do_register(connfd,tmp[1],tmp[2])
        elif tmp[0] == 'L':
            # "L name password"
            do_login(connfd,tmp[1],tmp[2])
        elif tmp[0] == 'Q':
            # "Q name word"
            do_query(connfd,tmp[1],tmp[2])

# 创建多进程并发模型
def main():
    # 创建套接字
    s = socket()
    s.bind(ADDR)
    s.listen(3)

    # 处理僵尸进程
    signal.signal(signal.SIGCHLD,signal.SIG_IGN)

    print("Listen the port 8000")
    while True:
        # 等待客户端链接
        c,addr = s.accept()
        print("Connect from",addr)

        # 为每个客户端创建进程
        p = Process(target=handle,args=(c,))
        p.start()



if __name__ == '__main__':
    main()