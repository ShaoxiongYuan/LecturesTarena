"""
epoll 方法实践
"""
from socket import *
from select import *


# 创建套接字作为监控对象
s = socket()
s.bind(("0.0.0.0",8888))
s.listen(5)

s.setblocking(False)

# 关注s
ep = epoll()
ep.register(s,EPOLLIN)

# 建立查找字典 时刻与register保持一直
fdmap = {s.fileno():s}

# 提交监控
while True:
    print("等待IO发生....")
    events = ep.poll()
    for fd,event in events:
        # 通过文件描述符值判定哪个IO就绪
        if fd == s.fileno():
            c,addr = fdmap[fd].accept()
            print("Connect from",addr)
            # 关注新的套接字对象
            c.setblocking(False)
            ep.register(c,EPOLLIN|EPOLLERR|EPOLLET) # 设置为边缘出发
            fdmap[c.fileno()] = c # 维护字典，与register保持一直
        # elif event & EPOLLIN:
        #     # 有客户端发送消息
        #     data = fdmap[fd].recv(1024).decode()
        #     if not data:
        #         ep.unregister(fd) # 取消关注
        #         fdmap[fd].close()
        #         del fdmap[fd]  # 从字典中删除
        #         continue
        #     print(data)
        #     fdmap[fd].send(b'OK')

