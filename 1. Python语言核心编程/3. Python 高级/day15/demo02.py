"""
    主动抛出异常
        快速传递错误信息
    # 11:25
"""


class AgeError(Exception):
    """
        封装数据
    """
    def __init__(self, message="", error_id=0, code=""):
        self.message = message
        self.error_id = error_id
        self.code = code

class Wife:
    def __init__(self, name="", age=0):
        self.name = name
        self.age = age

    @property
    def age(self):
        return self.__age

    @age.setter
    def age(self, value):
        if 22 <= value <= 32:
            self.__age = value
        else:
            # raise Exception("我不要")
            # raise AgeError("我不要",1001,"if 22 <= value <= 32")
            raise Exception("我不要",1001,"if 22 <= value <= 32")

# try:
#     w01 = Wife("双儿", 45)
# except AgeError as e:
#    print(e.error_id)
#    print(e.message)
#    print(e.code)

try:
    w01 = Wife("双儿", 45)
except Exception as e:
   print(e.args[0])
   print(e.args[1])
   print(e.args[2])