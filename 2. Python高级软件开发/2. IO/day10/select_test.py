"""
IO多路复用示例
"""
from socket import *
from select import select
from time import sleep

f = open('test.log')

s = socket()
s.bind(('0.0.0.0',8888))
s.listen(3)

sleep(5)
print("监控IO")


rs,ws,xs = select([f,s],[],[])
print("rlist:",rs)
print("wlist:",ws)
print("xlist:",xs)



