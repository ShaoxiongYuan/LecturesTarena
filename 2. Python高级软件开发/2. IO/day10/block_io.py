"""
非阻塞io演示
"""
from socket import *
from time import ctime,sleep


# tcp套接字
sockfd = socket()
sockfd.bind(('127.0.0.1',8800))
sockfd.listen(3)

# 将套接字设置为非阻塞
# sockfd.setblocking(False)

# 设置超时检测
sockfd.settimeout(2)

f = open("test.log",'a') # 日志文件

while True:
    print("Waiting for connect...")
    try:
        c,addr = sockfd.accept()
        print("Connect from",addr)
    except (BlockingIOError,timeout) as e:
        # 做一些事情， 和accept无关
        sleep(2)
        msg = "%s:%s\n"%(ctime(),e)
        f.write(msg)
        f.flush()
    else:
        data = c.recv(1024)
