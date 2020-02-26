"""
自定义进程类
"""

from multiprocessing import Process


# 自己定义一个进程类
class MyProcess(Process):
    def __init__(self,value):
        super().__init__() # 加载父类的init中的属性
        self.value = value

    # 充分发挥才能，实现你的进程事件
    def fun1(self):
        print("步骤一")

    def fun2(self):
        print("步骤二")

    # 功能启动函数
    def run(self):
        self.fun1()
        self.fun2()

p = MyProcess(2) # 定义进程对象
p.start() # 将run作为进程执行
p.join()
