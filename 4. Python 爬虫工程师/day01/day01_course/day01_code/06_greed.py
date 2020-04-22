"""
贪婪匹配和非贪婪匹配
"""
import re

html = """
<div><p>如果你为门中弟子伤她一分,我便屠你满门</p></div>
<div><p>如果你为天下人损她一毫,我便杀尽天下人</p></div>
"""

# 贪婪匹配
pattern = re.compile('<div><p>.*</p></div>',re.S)
r_list = pattern.findall(html)
print(r_list)

# 非贪婪匹配
pattern = re.compile('<div><p>(.*?)</p></div>',re.S)
r_list = pattern.findall(html)
print(r_list)













