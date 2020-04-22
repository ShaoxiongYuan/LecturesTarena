"""
params参数使用
"""
import requests

url = 'http://tieba.baidu.com/f?'
params = {'wd':'赵丽颖吧', 'pn':'50'}
headers = {'User-Agent' : 'Mozilla/5.0'}

html = requests.get(url=url, params=params, headers=headers).content.decode('utf-8', 'ignore')
print(html)










