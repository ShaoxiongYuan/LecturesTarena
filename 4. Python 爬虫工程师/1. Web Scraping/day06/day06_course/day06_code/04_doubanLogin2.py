"""
模拟登录豆瓣方法2
思路：
    1、先登录成功1次,抓取到Cookie
    2、Cookie处理成字典,然后利用 requests.get() 中的cookies参数实现模拟登录
"""
import requests

def login():
    url = 'https://www.douban.com/people/211922653/'
    headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36'}
    # F12抓包抓到的Cookie
    cook_str = 'll="118097"; bid=RXol6gm9_z8; _vwo_uuid_v2=D86CB471D688DB8877B43F317334F6048|560f53c46b5e7a6f5f2ac7fb1463f837; __utmz=30149280.1587803624.4.3.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; douban-profile-remind=1; push_noty_num=0; push_doumail_num=0; __utmc=30149280; _pk_ses.100001.8cb4=*; __utma=30149280.2126318535.1587721127.1587952263.1587958527.7; __utmt=1; dbcl2="211922653:YSkcfWeKUfQ"; ck=M01W; ap_v=0,6.0; __gads=ID=be0aec5eaed19432:T=1587958562:S=ALNI_MYseEt0nj3M1Xd_uO3mAgBK7QV1UA; __utmv=30149280.21192; _pk_id.100001.8cb4=672d49df57f4c9e4.1587866822.3.1587958825.1587952242.; __utmb=30149280.5.10.1587958527'
    # get_cookies(cook_str): 把字符串的cookie处理为字典
    cookies = get_cookies(cook_str)
    html = requests.get(url=url, headers=headers, cookies=cookies).text
    print(html)

def get_cookies(cook_str):
    """处理cookie为字典"""
    cookies = {}
    for kv in cook_str.split('; '):
        key = kv.split('=')[0]
        value = kv.split('=')[1]
        cookies[key] = value

    return cookies

login()








