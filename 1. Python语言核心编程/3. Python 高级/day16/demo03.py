"""
    生成器表达式
    # 做法有一个人完成，调用由另外一个人完成。使用生成器函数
    # 做法与调用由统一人完成,使用生成器表达式(优先).
    练习：exercise03
"""

list01 = [3, 42, 34, 435, 5, 64, 5, 67]

# list_result = [item for item in list01 if item > 10]
# for item in list_result:
#     print(item)

for item in (item for item in list01 if item > 10):
    print(item)

