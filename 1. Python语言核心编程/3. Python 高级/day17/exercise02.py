"""
    在不改变函数(进入后台,删除订单)的定义与调用基础上,
    为其增加新功能(验证权限)。
    要求：调试查看程序执行过程
"""


def verif_permissions(func):
    def wrapper(*args, **kwargs):
        print("验证权限")
        return func(*args, **kwargs)

    return wrapper


@verif_permissions
def enter_background(login_id, pwd):
    print(login_id, "进入后台")


@verif_permissions
def delete_order(id):
    print("删除%d订单")

enter_background("zs@qq.com", "1234")
delete_order(10010)
