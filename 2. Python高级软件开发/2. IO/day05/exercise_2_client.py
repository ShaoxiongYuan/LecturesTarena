"""
发送学生信息

从客户端输入一个学生的信息
id  name  age  score
将信息打包发送给服务端
"""

from socket import *
import struct

# 服务器地址
ADDR = ('127.0.0.1',8888)

# 数据格式
st = struct.Struct("i32sif")

# 使用udp进行网络传输
s = socket(AF_INET,SOCK_DGRAM)

while True:
    print("*******************************************")
    id = int(input("ID:"))
    name = input("Name:").encode()
    age = int(input("Age:"))
    score = float(input("Score:"))

    # 将数据打包发送
    data = st.pack(id,name,age,score)
    s.sendto(data,ADDR)

s.close()

