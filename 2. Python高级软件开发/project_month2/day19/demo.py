"""
二级界面
"""
def er():
    while True:
        print("=============命令选项==============")
        print("***         a         ***")
        print("***         b         ***")
        print("***         zhuxiao         ***")
        print("===================================")
        cmd = input("输入命令：")
        if cmd == 'a':
            pass
        elif cmd == 'b':
            pass
        elif cmd == 'c':
            break


while True:
    print("=============命令选项==============")
    print("***         login         ***")
    print("***         register        ***")
    print("***         exit        ***")
    print("===================================")
    cmd = input("输入命令：")
    if cmd == '1':
        er()
    elif cmd == '2':
        er()
    elif cmd == '3':
        break

