"""
功能扩展标志
"""

import re

s = """Hello 
北京
"""

# 只匹配ascii
# regex = re.compile(r"\w+",flags=re.A)

# 让.匹配换行
# regex = re.compile(r".+",flags=re.S)

# 忽略字母大小写
# regex = re.compile(r"[a-z]+",flags=re.I)

# 让^ $ 可以匹配每一行的开头结尾位置
regex = re.compile(r'^北京',flags=re.M | re.I)

l = regex.findall(s)
print(l)