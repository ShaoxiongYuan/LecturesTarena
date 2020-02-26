"""
    运算符重载
"""


class Vector2:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return "向量的x分量为%d,y分量为%d" % (self.x, self.y)

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    # 比较相同
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    # 比较大小
    def __lt__(self, other):
        return self.x < other.x

pos = Vector2(10, 25)
dir = Vector2(1, 0)
print(pos + dir)  # pos.__add__(dir)
pos += dir
print(pos)

# list01 = [10]
# list01 += [20]# 在原有对象中添加新元素（在原有可变对象上修改）
# print(list01)  # [10, 20]
#
# list02 = [10]
# list02 = list02 + [20] # 创建了新对象
# print(list02)  # [10, 20]


v01 = Vector2(100, 200)
v02 = Vector2(100, 200)
print(v01 == v02)  #
list_vectors = [
    Vector2(200, 200),
    Vector2(400, 400),
    Vector2(100, 100),
    Vector2(300, 300),
    Vector2(500, 500),
]
print(list_vectors.count(Vector2(100, 100)))
for item in sorted(list_vectors):
    print(item)
