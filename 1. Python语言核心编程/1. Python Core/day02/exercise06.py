"""
    古代的秤,一斤十六量.
    获取两,显示几斤零几两.
"""
weight_liang = int(input("请输入两："))
jin = weight_liang // 16
liang = weight_liang % 16
print(str(jin) + "斤零" + str(liang) + "两")
