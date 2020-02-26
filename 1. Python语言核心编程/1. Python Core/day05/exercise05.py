"""
    list01 = [4,5,65,76,7,8]
    在列表中找出最大值
    算法：
        假设第一个是最大值
        如果第二个大于假设的,则替换假设的.
        如果第三个大于假设的,则替换假设的.
        如果第四个大于假设的,则替换假设的.
        最后输出假设的(最大的)
"""
list01 = [4, 5, 65, 76, 7, 8]
max_value = list01[0]

for i in range(1, len(list01)):
    if max_value < list01[i]:
        max_value = list01[i]
print(max_value)
# for item in list01:
#     if max_value < item:
#         max_value = item
