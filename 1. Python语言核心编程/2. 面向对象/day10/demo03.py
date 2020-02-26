"""
    静态方法
        可以独立存在的工具函数
"""
class Vector2:
    """
        二维向量
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def get_right():
        return Vector2(0, 1)

    @staticmethod
    def get_up():
        return Vector2(-1, 0)

    @staticmethod
    def get_left():
        return Vector2(0, -1)

list01 = [
    ["00", "01", "02", "03"],
    ["10", "11", "12", "13"],
    ["20", "21", "22", "23"],
    ["30", "31", "32", "33"],
]

# 位置
pos = Vector2(1, 2)
# 方向
# right = Vector2(0, 1)
right = Vector2.get_right()
# 需求：沿着某个方向移动
pos.x += right.x
pos.y += right.y

print(pos.x,pos.y)

# 练习1:创建向上的静态方法
# 练习2:创建向左的静态方法
# 测试：让某个位置沿着该方向移动
up = Vector2.get_up()
pos.x += up.x
pos.y += up.y
print(pos.x,pos.y)












