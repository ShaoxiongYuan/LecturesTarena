"""
    课程3: ｐｙｔｈｏｎ高级
    程序结构
        包/模块
        导入是否成功的唯一条件：
            导入路径 + sys.path --> 真实路径
    异常处理
        现象：不再向下执行，而是不断向上返回.
        目的：错误流程 --> 正常流程
             保证程序按照既定流程执行
    迭代
        可迭代对象：__iter__
                 可以参与for
        迭代器:__next__
                获取获取下一个元素
    生成器
        本质：可迭代对象 +  迭代器
        作用：惰性操作/延迟操作(一次返回一个,不存储所有)
        语法：生成器函数、生成器表达式
        适用性：返回多个数据使用yield
                  单个       return
        用法：
            for          ---  惰性操作
            容器(生成器)   ---  立即操作

    函数式编程
        函数作为参数 --> 将核心逻辑传入通用方法中
            lambda --> 匿名方法(表达需要传递的核心逻辑)
        函数作为返回值 --> 逻辑连续
            装饰器　--> 拦截调用
"""


def func01():
    for i in range(10):
        yield i


# 因为具有__iter__,所以可以参与for
# for item in func01():
#     print(item)
print(tuple(func01()))
print(tuple(range(99999999999)))  # MemoryError

# 因为具有__next__,所以可以直接获取元素
iterator = func01()
print(iterator.__next__())
print(iterator.__next__())

# list01 = [10,20]
# print(list01.__next__())# 列表是可迭代对象，不是迭代器.
