"""
   Enclosing  外部嵌套作用域 ：函数嵌套。
"""

def func01():
    # 局部作用域
    a = 100
    # 外部嵌套作用域
    def func02():
        b = 200
        print("外部嵌套变量：",a)
    func02()

func01()



def func03():
    # 局部作用域
    a = 100
    # 外部嵌套作用域
    def func04():
        # a = 200 # 没有修改外部嵌套变量,而是创建了新局部变量
        nonlocal a # 通过nonlocal声明变量,可以修改外部嵌套变量
        a = 200
        print("外部嵌套变量：",a)

    func04()
    print("外部嵌套变量：", a)

func03()

