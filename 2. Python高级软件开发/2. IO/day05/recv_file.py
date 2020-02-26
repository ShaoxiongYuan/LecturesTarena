"""
上传头像  将一个图片从给客户端发送给服务端

思路：  客户端  读取头像图片内容--》通过网络发送给服务端
       服务端  从网络中接收内容--》写入到服务端的文件
"""

from socket import *

# 创造监听套接字
s = socket()
s.bind(('127.0.0.1',8889))
s.listen(3)

# 等待客户端链接
c,addr = s.accept()
print("Connect from",addr)


# 接收图片
f = open('gg.jpg','wb')

# 循环接收写入
while True:
    data = c.recv(1024)
    if data == "文件发送完毕".encode():
        print("接收完成")
        break
    f.write(data)

f.close()
c.close()
s.close()