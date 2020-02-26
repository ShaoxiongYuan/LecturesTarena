"""
字符串 和 字节串
"""

s = "hello world"
print(type(s))
print(s)

s = b"Hello world" # 普通字（英文字符）符串前面加b就变成了字节串
print(type(s))
print(s)


s = "你好"
print(s.encode())  # 将变量或者非英文字符串转换为字节串
print("武汉".encode())

s = "武汉".encode()
print(s.decode()) # 将字节串转换为字符串

