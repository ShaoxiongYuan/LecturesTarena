"""
udp客户端演示
重点代码
"""

from socket import *

# 服务端地址
ADDR = ('127.0.0.1',8888)

# 创建套接字
sockfd = socket(AF_INET,SOCK_DGRAM)

# 发送消息
while True:
    data = input(">>")
    if not data:
        break
    sockfd.sendto(data.encode(),ADDR)  # 字节串
    data,addr = sockfd.recvfrom(1024)
    print("服务端:",data.decode())

sockfd.close()