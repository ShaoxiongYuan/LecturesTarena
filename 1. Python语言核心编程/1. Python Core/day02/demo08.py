"""
    短路逻辑
        一但结果确定，后面的语句将不再执行。
        价值：尽量将耗时的判断放在后面
"""

# False  and ?
result = 1 > 2 and input("请输入") == "a"


# True  or ?
result = 1 < 2 or input("请输入") == "a"
# 终端中是否有"请输入"
