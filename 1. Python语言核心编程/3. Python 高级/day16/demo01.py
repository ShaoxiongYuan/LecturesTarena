"""
    内置生成器
        enumerate
"""
list01 = ["唐僧", "猪八戒", "悟空"]
dict01 = {"唐僧": 101, "八戒": 102, "悟空": 103}
for item in enumerate(list01):
    print(item)  # (索引,元素)

# 快捷键: itere + tab
for i, item in enumerate(list01):
    print(i, item)
