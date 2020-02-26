"""
regex1.py  re模块功能演示
"""

import re

# 目标字符串
s = "Alex:1996,Sunny:1998"
pattern = r"(\w+):(\d+)" # 正则表达式

# re模块调用
l = re.findall(pattern,s)
print(l)

# 正则表达式对象调用
regex = re.compile(pattern)
l = regex.findall(s,0,13)  # s[0:13] 作为匹配目标
print(l)

# 对目标字符串切割
l = re.split(r':|,',s,2)
print(l)


# 替换匹配到的内容
s = re.sub(r':','--',s,1)
print(s)








