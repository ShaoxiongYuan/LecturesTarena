"""
    在终端中循环录入商品信息(名称,单价)
    如果名称为空,停止录入.
    -- 打印所有商品信息(一行一个)
        格式：xxx的价格是yyy.
    -- 如果录入了游戏机,单独打印其价格.
"""
dict_commoditys = {}
while True:
    name = input("请输入商品名称：")
    if name == "":
        break
    price = float(input("请输入商品单价："))
    dict_commoditys[name] = price
for k, v in dict_commoditys.items():
    print("%s的价格是%f." % (k, v))
if "游戏机" in dict_commoditys:
    print(dict_commoditys["游戏机"])
