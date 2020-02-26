"""
    内置可重写函数
"""


class Wife:
    def __init__(self, name, face_score, money):
        self.name = name
        self.face_score = face_score
        self.money = money

    # 重写__str__作用是：定义当前对象转换的字符串格式
    #                  显示对象时使用（没有格式限制）
    def __str__(self):
        return "臣妾%s,颜值%d,存款%f." % (self.name, self.face_score, self.money)

    # 重写__repr__作用是：定义当前对象转换的字符串格式
    #                   克隆当前对象（python语法格式）
    def __repr__(self):
        return 'Wife("%s", %d, %f)' % (self.name, self.face_score, self.money)


w01 = Wife("双儿", 100, 50000)
message = w01.__str__()
print(message)  # <__main__.Wife object at 0x7ff8a3224cf8>
w02 = Wife("阿珂", 150, 20000)
print(w02)

str_code = w01.__repr__()
# eval:将字符串作为python代码执行
w03 = eval(str_code)  # 克隆老婆对象
w01.name = "双双"
print(w03.name)  #
