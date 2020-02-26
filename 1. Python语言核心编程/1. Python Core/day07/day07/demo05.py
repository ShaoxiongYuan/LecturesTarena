"""
    函数返回值
        语法

    函数创建者 给 调用者 传递的结果
"""

def func01():
    print("func01执行喽~")
    return 100 # return 数据

re = func01()

print(re)

def func02():
    print("func02执行喽~")
    # return  # return后没有数据或者没有return ,默认为None

re = func02()
print(re)

def func03():
    print("func03执行喽~")
    return # 退出函数
    print("func03又执行喽~")

func03()




