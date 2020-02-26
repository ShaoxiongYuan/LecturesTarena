# 练习1:["齐天大圣","八戒","唐三藏"]-->key:字符  vlaue:字符长度
#      {"齐天大圣":4,"八戒":2,"唐三藏":3}
list_names = ["齐天大圣", "八戒", "唐三藏"]
dict_names = {item: len(item) for item in list_names}
print(dict_names)

# 练习2:["张无忌","赵敏","周芷若"] [101,102,103]
#      {"张无忌":101,"赵敏":102,"周芷若":103}
list_names = ["张无忌", "赵敏", "周芷若"]
list_rooms = [101, 102, 103]
dict_infos = {list_names[i]: list_rooms[i] for i in range(len(list_names))}
print(dict_infos)

# 练习3：将练习2的字典,key 与 value 颠倒
#       {101:"张无忌",102:"赵敏",103:"周芷若"}
dict_infos2 = {v: k for k, v in dict_infos.items()}
print(dict_infos2)
