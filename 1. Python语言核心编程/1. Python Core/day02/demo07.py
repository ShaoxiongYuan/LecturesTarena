"""
    bool 运算
        bool类型
        比较运算符
        逻辑运算符
    练习:exercise09.py
"""
# 1. 布尔 bool 类型
# 命题：带有判断性质的陈述句
#      我是一个帅哥
#      真/对/成立    True
#      假/错/不成立  False

# 类型转换： bool(数据)  有值true  没值false

print(bool(0))  # False
print(bool(0.0))  # False
print(bool(""))  # False
print(bool(None))  # False

# 2. >  <   >=   <=  等于==  不等于!=
#  比较数值大小,结果是bool类型
data01 = 50
data02 = "50"
print(data01 == data02)  # false
print(50 != "50")  # false

# 3. 与and   或    非
# 判断多个命题关系

# 与and 现象：一假俱假  并且关系（必须满足所有条件）
print(True and True)  # True
print(False and True)  # False
print(True and False)  # False
print(False and False)  # False

# 或or 现象：一真俱真  或者关系（满足一个就行）
print(True or True)  # True
print(False or True)  # True
print(True or False)  # True
print(False or False)  # False

# 非  取反
print(not True)  # False
