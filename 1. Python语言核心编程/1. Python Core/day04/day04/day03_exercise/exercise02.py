"""
    4. 在终端中根据身高、体重显示身体情况.
        bmi:用体重(千克)除以身高(米)的平方
        小于18.5(不包含)  -->  体重过轻
        18.5 ~ 24(不包含)  -->  体重正常
        24 ～ 28(不包含)  -->  超重
        28 ～ 30(不包含)  -->  I度肥胖
        30 ～ 40(不包含)  -->  II度肥胖
        30 ～ -->  III度肥胖
"""
height = float(input("请输入身高(米)："))
weight = float(input("请输入体重(千克)："))
bmi = weight / height ** 2

if bmi < 18.5:
    print("体重过轻")
elif bmi < 24:
    print("体重正常")
elif bmi < 28:
    print("超重")
elif bmi < 30:
    print("I度肥胖")
elif bmi < 40:
    print("II度肥胖")
else:
    print("III度肥胖")
