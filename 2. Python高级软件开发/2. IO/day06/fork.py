import os
from time import sleep

pid = os.fork()

if pid < 0:
    print("进程创建失败")
elif pid == 0:
    sleep(3)
    print("新的进程")
else:
    sleep(4)
    print("原来的进程")

print("实验完毕")
