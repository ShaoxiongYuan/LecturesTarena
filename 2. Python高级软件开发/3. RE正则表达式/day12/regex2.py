"""
正则表达式模块演示
"""
import re

s = "2020年2月19日"
pattern = r'\d+'

# 正则表达式匹配，获取迭代对象
# it = re.finditer(pattern,s)
#
# for i in it:
#     print(i.group())  # 每个match对象一定对应一处结果

# 匹配目标字符串开始位置
obj = re.match(pattern,s)
print(obj.group())

# 匹配第一处内容
obj = re.search(pattern,s)
print(obj.group())
