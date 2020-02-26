"""
数据打包示例
"""

import struct

# 数据 1 b“Lily” 168 92.5

# 确定格式对象
st = struct.Struct("i4sif")

# 数据打包
data = st.pack(1,b'li',168,92.5)
print(data)

data = st.unpack(data)
print(data)
print(data[1].decode().strip("\x00")) # 去除字节穿两测的\x00

# struct 直接调用
data = struct.pack("i4sif",1,b'li',168,92.5)
print(data)
data = struct.unpack("i4sif",data)
print(data)