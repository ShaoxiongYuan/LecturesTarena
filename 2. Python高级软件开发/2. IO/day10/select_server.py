from socket import *
from select import select

"""
可否同时监控多个客户端和一个监听套接字
有客户端链接 --》 监听套接字
客户端给我发消息 --》 对应的客户端套接字
"""

s = socket()
s.bind(('0.0.0.0',8888))
s.listen(3)

rlist = [s]
c,addr = s.accept()
rlist.append(c)
c,addr = s.accept()
rlist.append(c)

while True:
    rs,ws,xs = select(rlist,[],[])
    print(rs)