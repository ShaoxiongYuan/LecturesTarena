"""
输入搜索关键字,把响应内容保存到本地文件
使用quote()方法实现
"""
import requests
from urllib import parse

# 1、拼接URL地址 ：http://www.baidu.com/s?wd=xxx
keyword = input('请输入关键字:')
params = parse.quote(keyword)
url = 'http://www.baidu.com/s?wd={}'.format(params)
headers = { 'User-Agent' : 'Mozilla/5.0' }

# 2、发请求获取响应内容
html = requests.get(url=url, headers=headers).text

# 3、保存到本地文件
filename = '{}.html'.format(keyword)
with open(filename, 'w', encoding='utf-8') as f:
    f.write(html)












