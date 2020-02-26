"""
thread_lock.py lock方法解决同步互斥
"""

from threading import Thread,Lock

a = b = 0
lock = Lock() # 创建锁对象

# 线程函数
def value():
    while True:
        lock.acquire()
        if a != b:
            print("a = %d,b = %d"%(a,b))
        lock.release()

t = Thread(target=value)
t.start()

while True:
    lock.acquire() # 上锁
    a += 1
    b += 1
    lock.release() # 解锁

    # with lock:  #上锁
    #     a += 1
    #     b += 1
                # 语句块结束解锁

t.join()