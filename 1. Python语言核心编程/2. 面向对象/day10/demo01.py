"""
    实例成员
        实例变量：对象不同的数据
        实例方法：对象相同行为
        核心逻辑：
           对象.?
"""


# 1. 标准写法
class Wife:
    def __init__(self, name):
        # 创建实例变量：对象.变量名 = 数据
        self.name = name

    def print_self(self):
        print("我叫：", self.name)

    def play(self):
        print("在玩耍")
        # 对象.方法名()
        self.print_self()


w01 = Wife("双儿")
w02 = Wife("阿珂")

# 读取实例变量：?  = 对象.变量名
print(w01.name)

# 内置实例变量
# -- 获取当前对象所有实例变量
print(w01.__dict__)

w01.play()
w02.play()

# 2. 不建议(实例变量应该由类的定义者决定)
# class Wife:
#     pass
#
# w01 = Wife()
# # 创建实例变量：对象.变量名 = 数据
# w01.name = "双儿"
# # 读取实例变量：?  = 对象.变量名
# print(w01.name)

# 3. 不建议(实例变量应该直接在init中定义)
# class Wife:
#     def __init__(self):
#         pass
#
#     def func01(self,name):
#         # 创建实例变量：对象.变量名 = 数据
#         self.name = name
#
# w01 = Wife()
# w01.func01("双儿")
# # 读取实例变量：?  = 对象.变量名
# print(w01.name)


dict01 = {}
dict01["name"] = "双儿"
dict01["name"] = "双双"
