"""
    字符串字面值
    练习:exercise06.py
"""

# 1. 双引号
name01 = "悟空"
#    单引号
name02 = '悟空'

#    三引号(可见即所得)
name03 = '''悟空'''
name04 = """悟空"""

message = '我叫"齐天大圣".'
message = "我叫'齐天大圣'."
message = """我叫'齐"天大"圣'."""

# 2. 转义字符：改变字符的原始含义。
# \"  \'   \\
message = "我叫\"齐天大圣\"."
url = "C:\\arogram Files\\bntel\cedia SDK"
# 原始字符串：没有转义字符
url = r"C:\arogram Files\bntel\cedia SDK"
print(url)

# \t 水平制表格
print("我爱\t编程.")
# \n 换行
print("我爱\n编程.")

# 3. 字符串格式化
name = "qtx"
age = 18
score = 95.12345
print("我的名字是%s,年龄是%d,成绩是%.3f" % (name, age, score))
