"""
让西刺代理网站封掉我！
"""
import requests

url = 'https://www.xicidaili.com/'
headers = {'User-Agent':'Mozilla/5.0'}
proxies = {
    'http' : 'http://309435365:szayclhp@42.51.205.96:16819',
    'https' : 'https://309435365:szayclhp@42.51.205.96:16819',
}

html = requests.get(url=url, proxies=proxies, headers=headers).text
print(html)










