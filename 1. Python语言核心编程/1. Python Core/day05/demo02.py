"""
    深拷贝

    优点：
        对其中之一的修改，
        绝对不影响另外一个.
    缺点：
        比较占用内存
"""
# 准备深拷贝的工具
import copy

list01 = [
    [1,2,3],
    [4,5,6]
]
# list02 =list01[:]
# 深拷贝列表
list02 = copy.deepcopy(list01)
list01[0] = "一"
list01[1][0] = "二"
print(list02)#
