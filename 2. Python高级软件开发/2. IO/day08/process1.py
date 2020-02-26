"""
multiprocessing 模块创建进程
【1】 将需要子进程执行的事件封装为函数
【2】 通过模块的Process类创建进程对象，关联函数
【3】 通过进程对象调用start启动进程
【4】 通过进程对象调用join回收进程
"""
from time import sleep
import multiprocessing


a = 1

# 进程执行函数
def fun():
    print("开始一个进程")
    sleep(5)
    global a
    print("a = ",a)
    a = 1000
    print("子进程执行结束")


# 创建进程对象
p = multiprocessing.Process(target=fun)

# 启动进程 此时进程诞生 执行函数作为进程的执行内容
p.start()

# 父进程做的
sleep(3)
print("父进程事件")

# 回收进程，防止僵尸进程
p.join(1)
print("==================================")
print("a:",a)


"""
p = os.fork()
if pid == 0:
    fun()
    os._exit()
else:
    os.wait()
"""