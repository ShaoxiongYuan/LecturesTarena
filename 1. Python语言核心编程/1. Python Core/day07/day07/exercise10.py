# 练习:定义函数,在终端中根据边长打印矩形
#  day05/day04_exercse/exercise01

def print_rectangle(lenght,char):
    """
        打印矩形
    :param lenght:int类型 边长
    """
    print(char * lenght)
    for __ in range(lenght - 2):
        print(char + " " * (lenght - 2) + char)
    print(char * lenght)

# 测试
print_rectangle(5,"*")
print_rectangle(10,"$")