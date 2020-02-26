"""
    创建1个狗类
        数据：爱称,年龄,颜色,品种
        行为：吃、坐下

    创建2个狗对象,调用相应的方法
    画出代码内存图
"""


class Dog:
    """
        狗
    """

    def __init__(self, name, age=0, color="", breed=""):
        self.pet_name = name
        self.age = age
        self.color = color
        self.breed = breed

    def eat(self):
        print(self.pet_name, "在吃")

    def sit(self):
        print(self.pet_name, "坐下")


mx = Dog("米咻", 4, "黄", "拉布拉多")
hm = Dog("黑米", 3, "黑", "拉布拉多")
mx.sit()
hm.sit()

# 练习1：
mx = Dog("米咻", 4, "黄", "拉布拉多")
d01 = mx
d01.pet_name = "咻咻"
mx.age = 5
print(d01.pet_name)
print(d01.age)
print(mx.pet_name)
print(mx.age)

# 练习2：
# 17:00
def func01(p1,p2):
    p1 = Dog("小白", 5)
    p2.pet_name = "小黑"
    p2.age = 6

mx = Dog("米咻", 4)
hm = Dog("黑米", 3)
func01(mx,hm)
print(mx.pet_name,mx.age)#?
print(hm.pet_name,hm.age)#"小黑"   6















