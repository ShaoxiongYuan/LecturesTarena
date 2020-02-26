"""
seek.py  文件偏移量展示

open 文件偏移量初始化
r w 打开文件偏移量在开头
a 打开文件偏移量在结尾
"""

# 读写方式
f = open('file','wb+')

f.write("春天来了".encode())
f.flush()

print("当前文件偏移量位置:",f.tell())

# 移动文件偏移量
f.seek(-6,2) # 以开头为基准,向后移动0 字节

data = f.read()
print(data.decode())

f.close()