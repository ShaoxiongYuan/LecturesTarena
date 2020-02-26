"""
进程对象属性
"""

from multiprocessing import Process
import time

def fun():
    for i in range(3):
        print(time.ctime())
        time.sleep(2)

# 创建进程对象
p = Process(target=fun,name = "Tarena")

p.daemon = True # 父进程退出时，它创建的这个子进程也退出

p.start()

# p.name = "Tedu"
print("Name:",p.name)
print("is alive:",p.is_alive())
print("PID:",p.pid)
time.sleep(1)