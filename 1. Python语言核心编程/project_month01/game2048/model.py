"""
    数据模型
"""


# 元组　(2,3)　　  　位置(2,3)
# 元组[0] --> 2     位置.r  --> 2
class Location:
    """
        位置
    """

    def __init__(self, r=0, c=0):
        self.r = r
        self.c = c
