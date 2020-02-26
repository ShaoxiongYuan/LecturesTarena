"""
1. 执行两个函数事件，计算执行时间
2. 使用两个进程同时执行这两个事件再计算时间
"""

def sum():
    result = 0
    for i in range(99999999):
        result += i
    print("结果：%.2f万",result/10000)

def write():
    with open("file.txt") as f:
        for i in range(999999):
            f.write("Hello world\n")