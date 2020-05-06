"""
多线程使用示例
"""
from threading import Thread

def spider():
    """线程事件函数"""
    print('I am spider man')

t_list = []

for i in range(5):
    t = Thread(target=spider)
    t_list.append(t)
    t.start()

for t in t_list:
    t.join()












