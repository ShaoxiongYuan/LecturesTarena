"""
免费代理IP测试
"""
import requests

url = 'http://httpbin.org/get'
headers = { 'User-Agent' : 'Mozilla/5.0' }
proxies = {
    'http':'http://182.32.234.18:9999',
    'https':'https://182.32.234.18:9999'
}
html = requests.get(url=url, proxies=proxies, headers=headers).text
# html: origin对应的IP到底是什么
print(html)










