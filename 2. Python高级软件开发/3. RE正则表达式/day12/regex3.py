"""
演示match对象的作用
"""

import re

pattern = r"(ab)cd(?P<dog>ef)"

regex = re.compile(pattern)

obj = regex.search("abcdefghki")

print(obj.span()) # 匹配到的内容在目标字符串中的位置 s[0:6]

print(obj.groupdict()) # 捕获组组名与其对应内容的字典

print(obj.group())
print(obj.group(1))
print(obj.group('dog'))
