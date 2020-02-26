"""
演示僵尸进程产生
"""

import os,sys
import time
import signal

# 忽略子进程的退出行为
signal.signal(signal.SIGCHLD,signal.SIG_IGN)


pid = os.fork()

if pid < 0:
    print("Error")
elif pid == 0:
    print("子进程：",os.getpid())
    sys.exit(2)
else:
    # 父进程如果长期不退出，那么有可能会有大量的僵尸存留
    print("父进程")

    # 阻塞等待子进程退出，进行回收
    # pid,status = os.wait()
    # print("子进程PID：",pid)
    # print("子进程退出状态:",status/256) # 退出状态 * 256

    input()
