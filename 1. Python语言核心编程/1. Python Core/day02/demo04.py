"""
    数据类型
"""

# 变量（左边）没有类型,但是关联的对象(右边)有类型
data01 = 100
data01 = "悟空"
print(type(data01))


# 1. 空None类型
# -- 占位
name = None
# -- 解除与对象的关联
score = 100
score = None

# 2. 整形(整数)int类型
# -- 十进制：逢十进一 0 1 2 3 4 ..9  进位 10
data01 = 10
# -- 二进制：逢二进一 0 1 进位 10  11   100   101   110
data02 = 0b10
# -- 八进制:逢八进一 0 1 2 ...7 进位 10  11
data03 = 0o10
# -- 十六进制:逢十六进一 0 1 2 ...9  a(10)..f(15) 进位 10   11 ..
data04 = 0x10

# 3. 浮点型(小数)float类型
number01 =1.5
print(number01)
number02 =15e-1
number03 = 0.00000000001# 1e-11
print(number03)

# 4. 字符串str类型
message = "我爱编程"
message = "1.5"

