# 01_percetron.py
# 自定义感知机

# 逻辑和
def AND(x1, x2):
    w1, w2 = 0.5, 0.5  # 权重
    theta = 0.7  # 阈值
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    else:
        return 1

def OR(x1, x2):
    w1, w2 = 0.5, 0.5  # 权重
    theta = 0.2  # 阈值
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    else:
        return 1

# 异或
def XOR(x1, x2):
    s1 = not AND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(AND(1, 1))  # 1
print(AND(0, 1))  # 0
print(AND(0, 0))  # 0
print("")
print(OR(1, 1))  # 1
print(OR(0, 1))  # 1
print(OR(0, 0))  # 0
print("")
print(XOR(1, 1))  # 0
print(XOR(0, 0))  # 0
print(XOR(0, 1))  # 1
print(XOR(1, 0))  # 1

