"""
with.py
with语句块测试
"""

with open("file") as f:  # 打开一个文件
    data = f.read()
    print(data)

# with语句块结束，f自动销毁
