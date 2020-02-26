"""
    一张纸的厚度是0.01毫米
    请计算,对折多少次超过珠穆朗玛峰(8844.43米).
"""

# 数据：厚度、次数
# thickness = 0.01 / 1000
# 算法：*=2   += 1

thickness = 1e-5
count = 0
while thickness <= 8844.43:
    thickness *= 2
    count += 1
    print("第" + str(count) + "次对折后的厚度是" + str(thickness))
print(count)
