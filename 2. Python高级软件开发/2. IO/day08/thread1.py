"""
thread1.py  创建线程演示
步骤 ：同Ｐｒｏｃｅｓｓ用法
1. 封装线程函数
2. 创建线程对象
3. 启动线程
4. 回收线程
"""

from threading import Thread
from time import sleep
import os

a = 1

# 线程函数
def music():
    for i in range(3):
        sleep(2)
        print(os.getpid(),"播放：黄河大合唱")
    global a
    print('a = ',a)
    a = 1000

# 创建线程对象
t = Thread(target=music)

t.start() # 启动线程 将music函数作为线程执行内容

for i in range(4):
    sleep(1)
    print(os.getpid(),"播放:葫芦娃")

t.join() # 回收线程

print("a:",a)
