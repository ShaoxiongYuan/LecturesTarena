"""
线程效率测试
"""
import time
from threading import Thread

def count(x,y):
    c = 0
    while c < 7000000:
        x += 1
        y += 1
        c += 1

tm = time.time()

# 用时： 8.73266339302063
# for i in range(10):
#     count(1,1)

# 用时： 7.338038206100464
jobs = []
for i in range(10):
    t = Thread(target=count,args = (1,1))
    jobs.append(t)
    t.start()

for i in jobs:
    i.join()

print("用时：",time.time() - tm)


