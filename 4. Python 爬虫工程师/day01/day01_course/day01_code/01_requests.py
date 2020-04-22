"""
访问京东官网,并拿到响应内容
"""
import requests

# 1、get()方法得到的为 响应对象
response = requests.get(url='https://www.jd.com/')
# 2、text属性：获取响应内容（字符串）
html = response.text
# 3、content属性：获取响应内容（字节串bytes）- 图片、音频、视频、文件 ... ...
html = response.content
# 4、status_code：HTTP响应码
code = response.status_code
# 5、url：实际返回数据的URL地址
url = response.url
print(url)














