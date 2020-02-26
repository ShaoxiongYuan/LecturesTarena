"""
套接字属性介绍
"""

from socket import *

s = socket()  # 创建套接字

s.bind(("192.168.31.196",8899))
s.listen(3)
c,addr = s.accept()

print("地址类型：", s.family)
print("套接字类型：", s.type)
print("绑定的地址：",s.getsockname())
print("文件描述符:",s.fileno())
print("连接的客户端地址：",c.getpeername())

