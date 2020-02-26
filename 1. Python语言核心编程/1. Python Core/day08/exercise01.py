# 1.
def func01(p1,p2):
    p1= ["孙悟空"]
    p2[0] = ["猪八戒"]

a = ["悟空"]
b = ["八戒"]
func01(a,b)
print(a)# ['悟空']
print(b)# [['猪八戒']]

# 2.
def func02(p1,p2):
    p2[:] = "孙悟空"
    p1[:] = p1[::-1]

a = [10,20,30]
b = ["a","b","c"]
func02(a,b)
print(a)
print(b)


