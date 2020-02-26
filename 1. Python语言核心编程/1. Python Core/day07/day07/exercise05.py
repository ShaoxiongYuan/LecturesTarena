"""
    思想：
        将第一位元素,与后面元素进行比较,发现更小的则交换
           二
           三
           ...
"""

list01 = [4, 54, 5, 6, 7, 8, 3]

# list01[0]  list01[1]
# list01[0]  list01[2]
# list01[0]  list01[3]
# list01[0]  list01[4]
# for i in range(1,len(list01)):
#     # list01[0] 与 list01[i] 进行比较
#     pass

# list01[1]  list01[2]
# list01[1]  list01[3]
# list01[1]  list01[4]
# list01[1]  list01[5]
# for i in range(2, len(list01)):
#     # list01[1] 与 list01[i] 进行比较
#     pass

for r in range(len(list01) - 1):  # 0       1
    for c in range(r + 1, len(list01)):  # 123..   234..
        if list01[r] > list01[c]:
            list01[r], list01[c] = list01[c], list01[r]

print(list01)
