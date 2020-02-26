"""
    身份运算符
"""
list01 = [100]
list02 = [100]

# 两个变量指向的对象是否同一个
print(list01 is list02)# False

list03 = list01
print(list03 is list01)# True

# 原理：判断变量存储的地址是否相同
print(id(list01))# 140235347048584
print(id(list02))# 140235357765192
print(id(list03))# 140235347048584

