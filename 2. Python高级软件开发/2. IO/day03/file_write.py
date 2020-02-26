"""
file_write.py
文件的写操作
"""

# 打开文件
# f = open('file','w')
f = open('file','a') # 追加

# 写文件
# f.write("Hello,死鬼\n")
# n = f.write("哎呀，干啥\n")
# print("写入了:",n)

# 写入多次
l = ['你好\n','hello\n','hi\n']
f.writelines(l)


# 关闭文件 f不存在了
f.close()
