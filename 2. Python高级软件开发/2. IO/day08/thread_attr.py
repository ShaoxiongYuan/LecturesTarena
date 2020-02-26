"""
线程属性
"""

from threading import Thread
from time import sleep

def fun():
    sleep(3)
    print("线程属性测试")

t = Thread(target = fun)

# 在start前设置,分支线程随主线程结束而结束
t.setDaemon(True)

t.start()

t.setName("Tedu")
print("Name:",t.getName())

print("is alive:",t.is_alive())
print("daemon:",t.isDaemon())