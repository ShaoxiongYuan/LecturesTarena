"""
要求: 1. 向一个文件中每隔2秒写入一行内容 :
　　　　　　
　　　　　　　1.  2020-01-01 12:12:12
　　　　　　　2.  2020-01-01 12:12:14
　　　　　　　3.  2020-01-01 12:12:16
　　　　　　　4.  2020-01-01 12:12:18
　　　　　　　5.  2020-01-01 12:12:28
　　　　　　　　　　

      2. 打开文件随时可以看到已经写入的内容
      3. 当程序终止后，重新启动后继续写并且行号可以衔接

温馨提示：　import time
           time.localtime()
           time.sleep()
"""

import time

f = open('file','a+') # 确保每次启动继续写

# 先判断有多少行
f.seek(0,0) #　把文件偏移量移动到开头
n = 1   # 应该等于　行数+1
for line in f:
    n += 1

while True:
    time.sleep(2)  # 2s间隔
    s = "%d.  %s\n"%(n,time.ctime())
    f.write(s)
    f.flush()
    n += 1

f.close()













