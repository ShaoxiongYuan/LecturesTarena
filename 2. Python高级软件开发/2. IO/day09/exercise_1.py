"""
作业：  使用多线程拷贝一个目录，要求同时拷贝该目录下的这些文件

     比如 原目录 --》 os.listdir()（10个文件）
         os.mkdir()  --》 新目录
         创建10个线程，同时拷贝这些文件，将这些文件拷贝到新目录

1. 拷贝文件怎么拷贝
2. 查看一下目录中有多少文件，就创建多少线程
3. 将文件从old_dir --> new_dir
"""

import os
from threading import Thread

old_dir = input("要拷贝哪个目录：")

if old_dir[-1] == '/':
    old_dir = old_dir[:-1]
new_dir = old_dir + '-备份'

try:
    file_list = os.listdir(old_dir)  # 文件列表
except:
    os._exit(0)

os.mkdir(new_dir)

# 拷贝文件
def copy_file(file):
    fr = open(old_dir+'/'+file,'rb')
    fw= open(new_dir+'/'+file,'wb')
    # 开始拷贝
    while True:
        data = fr.read(1024)
        if not data:
            break
        fw.write(data)
    fr.close()
    fw.close()



# 根据文件多少创建线程
jobs = []
for file in file_list:
    t = Thread(target=copy_file,args=(file,))
    jobs.append(t)
    t.start()

for i in jobs:
    i.join()


