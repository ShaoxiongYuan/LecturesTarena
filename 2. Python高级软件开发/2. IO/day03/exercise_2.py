"""
* 重点代码

编写一个函数 完成对文件的复制工作
*  将file1 复制一份存为file2

  file1 : 传入你要复制的文件
  file2:  传入你要将复制的文件存为什么

思路: 1. 打开file1 和file2
     2. 从file1中读取出内容，将这些内容写入到file2中
     3. 如果file1很大应该循环的进行读写

"""

def copy(file1,file2):
    fr = open(file1,'rb')
    fw = open(file2,'wb')

    while True:
        data = fr.read(1024)  # 从源文件读
        if not data:
            break
        fw.write(data) # 写入新的文件

    fr.close()
    fw.close()

copy("../day02/2.txt",'2.txt')