"""
http基本演示
"""
from socket import *

# 创建套接字
s = socket()
s.bind(('127.0.0.1',8000))
s.listen(3)

# 等待客户端（浏览器）链接
c,addr = s.accept()
print("Connect from",addr)

# http请求
data = c.recv(2048)
print(data.decode())

# 发送数据给浏览器
# data = """HTTP/1.1 404 Not Found
# Content-Type:text/html
#
# Hello World
# """

data = "HTTP/1.1 200 OK\r\n"
data += "Content-Type:text/html\r\n"
data += "\r\n"
data += "Hello world"


c.send(data.encode())

c.close()
s.close()