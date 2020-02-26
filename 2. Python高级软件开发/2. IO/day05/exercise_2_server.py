"""
使用udp完成

从客户端输入一个学生的信息
id  name  age  score
将信息打包发送给服务端

服务端收到信息之后如果成绩大于90则将其信息写入一个文件中，每个学生信息写一行

思路： udp传输结构
      打包结构
      如何写入文件
"""

from  socket import *
import struct

# 数据结构
st = struct.Struct("i32sif")

# udp套接字
s = socket(AF_INET,SOCK_DGRAM)
s.bind(('0.0.0.0',8888))

# 要写入的文件
f = open('student.txt','a')

# 循环接收内容
while True:
    data,addr = s.recvfrom(1024)
    # info --> (1,'Tom',12,98)
    info = st.unpack(data)
    # 分数大于90则写入文件
    if info[-1] >= 90:
        id,name,age,score = info
        name = name.decode().strip('\x00')
        stu = "%d  %s  %d   %.2f\n"%(id,name,age,score)
        f.write(stu)
        f.flush()



