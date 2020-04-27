"""
向测试网站：http://httpbin.org/get 发出请求,确认请求头中的User-Agent
"""
import requests

url = 'https://www.jd.com/'
html = requests.get(url=url, headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/14.0.835.163 Safari/535.1'}).text
print(html)







