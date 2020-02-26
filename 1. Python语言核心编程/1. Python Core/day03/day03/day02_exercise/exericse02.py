"""
4. 温度转换器
    公式：摄氏度 =  (华氏度- 32) 除以 1.8
    已知：已知摄氏度
    计算：华氏度
    格式：xx摄氏度是xx华氏度

    摄氏度 * 1.8 + 32=  华氏度
"""
centigrade = input("请输入摄氏度：")

fahrenheit = float(centigrade) * 1.8 + 32

print(centigrade + "摄氏度是"+str(fahrenheit)+"华氏度")






