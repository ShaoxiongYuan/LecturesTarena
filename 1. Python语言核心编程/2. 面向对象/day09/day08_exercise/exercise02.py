"""
    str01 = " 校 训：自 强不息,厚 德载 物 "
        -- 查找空格数量
        -- 删除前后空格
        -- 删除所有空格
        -- 查找"不息"的位置
"""
str01 = " 校 训：自 强不息,厚 德载 物 "
print(str01.count(" "))
print(str01.strip(" "))
print(str01.replace(" ",""))# 替换
print(str01.find("不息"))#