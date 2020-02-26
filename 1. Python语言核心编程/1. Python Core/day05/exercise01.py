"""
    列表内存图练习
"""
# 1.
list01 = ["谢逊"]
list02 = list01
list02.append("翠山")
list01.insert(0,"殷素素")
del list02[1]
print(list01)# ?
print(list02)# ?

# 2.
list01 = ["谢逊","翠山"]
list02 = list01[:]
list02[0] = "金毛狮王"
print(list01)# ?

# 3.
list01 = ["谢逊","翠山"]
list02 = list01
list02[0:1] = "金毛狮王"
print(list01)# ?


# 4.
list01 = ["谢逊",["翠山","殷素素"],["张无忌"],"赵敏"]
list02 = list01
list02[0] = ["金毛","狮王"]
list02[1][0] = "张翠山"
print(list01)# ?

# 5.
list01 = [
    [1,2,3],
    [4,5,6]
]
list02 = list01[::-1]
# 影响list02
list01[0][0] = "一"
# 不影响list02
list01[1] = [7,8,9]
print(list02)#















