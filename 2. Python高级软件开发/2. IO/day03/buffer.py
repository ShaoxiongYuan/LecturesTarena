"""
buffer.py  缓冲区演示

刷新缓冲区条件：
1.  缓冲区满了
2.  文件close()关闭
3.  程序结束
4.  调用flush()函数
"""

# f = open('file','w')

# f = open("file",'w',1) # 1 行缓冲,遇到\n会刷新缓冲区

f = open("file",'wb',10) # >1 指明缓冲区大小 (二进制方式)

while True:
    data = input(">>")
    if not data:
        break
    f.write(b"hello world")
    # f.flush() # 刷新缓冲区

f.close()