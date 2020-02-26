"""
    画出下列代码内存图

"""
# 1.
list01 = ["悟空"]
print(id(list01))
list01 += ["八戒"]
print(id(list01))
print(list01)  # ['悟空', '八戒']

tuple01 = ("悟空",)
print(id(tuple01))
tuple01 += ("八戒",)
print(id(tuple01))
print(tuple01)  # ('悟空', '八戒')
