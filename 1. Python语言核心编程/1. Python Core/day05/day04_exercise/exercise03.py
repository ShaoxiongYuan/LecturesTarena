"""
5. （扩展）一个小球从100米高度落下,每次弹回原高度一半(最小弹起高度0.01m)
   请计算：
   -- 总共弹起多少次？
   -- 整个过程走了多少米？
"""

# 数据：高度
# 算法：减半

height = 100
count = 0
distance = height
# 如果弹起来的高度(弹之前/2) 大于 最小弹起高度:则弹
while height / 2 > 0.01:
    # 弹
    height /= 2
    count += 1
    print("第%d次弹起来的高度是%f." % (count, height))
    distance += height * 2 # 累加起、落距离

print("总共弹起%d次." % count)
print("整个过程走了%.2f米." % distance)
