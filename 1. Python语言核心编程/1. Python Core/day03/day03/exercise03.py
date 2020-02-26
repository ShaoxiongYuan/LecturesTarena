"""
    录入4个同学体重,
    打印最重的体重。
    算法：
        假设第一个是最大值
        如果第二个大于假设的,则替换假设的.
        如果第三个大于假设的,则替换假设的.
        如果第四个大于假设的,则替换假设的.
        最后输出假设的(最大的)
"""
weight01 = float(input("请输入第一个同学体重："))
weight02 = float(input("请输入第一个同学体重："))
weight03 = float(input("请输入第一个同学体重："))
weight04 = float(input("请输入第一个同学体重："))
max_value = weight01
if max_value < weight02:
    max_value = weight02
if max_value < weight03:
    max_value = weight03
if max_value < weight04:
    max_value = weight04
print(max_value)
