"""
    将英文语句进行反转
    How are you   -->  you are How
"""
message = "How are you"
list_temp = message.split(" ")
str_result = " ".join(list_temp[::-1])
print(str_result)
