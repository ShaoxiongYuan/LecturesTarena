"""
    调用以下函数：
"""


def func01(*args, **kwargs):  # 万能形参
    print(args)
    print(kwargs)


func01(34, 234, 534, 65, 456, 467, 5, a=1, b=2, c=3)
func01()

def func02(a, b=0, *args, c=0, d=0, **kwargs):
    print(a)
    print(b)
    print(args)
    print(c)
    print(d)
    print(kwargs)

func02(1, 10, "a", "b", "c", c="C", d="D",e = "E",f ="F")
func02(1)
