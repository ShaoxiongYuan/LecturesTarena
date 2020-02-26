"""
使用input输入一个端口名称，匹配到其对应的  address is 什么

段落之间有空行
每段的收个单词是端口名称

思路： 先找到对应的那一段文字
"""

import re

port = input("端口:")

f = open('exc.txt')

while True:
    # 找到目标段
    data = ''
    for line in f:
        if line == '\n':
            # 一段结束
            break
        data += line
    if not data:
        # 文件结尾
        break

    # 匹配首单词
    obj = re.match(r"\S+",data)
    # 确定是否为目标段
    if port == obj.group():
        pattern = r"[0-9a-f]{4}\.[0-9a-f]{4}\.[0-9a-f]{4}"
        obj = re.search(pattern,data)
        print(obj.group())
        break



