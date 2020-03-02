import json

d = {'a':1,'b':"hello"}

data = json.dumps(d)  # 将字典转换为json字符串

data = json.loads(data)  # 将json字符串转换为字典
print(type(data))