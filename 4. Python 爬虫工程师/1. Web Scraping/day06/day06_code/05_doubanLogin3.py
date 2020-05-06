"""
requests.session() 实现模拟登录
思路：
    1、实例化session对象
    2、登录网站 - s.post()
    3、抓取数据 - s.get()
"""
import requests

# 实例化session对象
s = requests.session()

def login():
    # 1、登录
    post_url = 'https://accounts.douban.com/j/mobile/login/basic'
    data = {
        'ck' : '',
        'name' : '15110225726',
        'password' : 'zhanshen001',
        'remember' : 'false',
        'ticket' : '',
    }
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36',
        'Cookie' : 'll="118097"; bid=RXol6gm9_z8; _vwo_uuid_v2=D86CB471D688DB8877B43F317334F6048|560f53c46b5e7a6f5f2ac7fb1463f837; __utmz=30149280.1587803624.4.3.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; douban-profile-remind=1; push_noty_num=0; push_doumail_num=0; apiKey=; __utmc=30149280; last_login_way=account; __gads=ID=be0aec5eaed19432:T=1587958562:S=ALNI_MYseEt0nj3M1Xd_uO3mAgBK7QV1UA; __utmv=30149280.21192; _pk_ref.100001.2fad=%5B%22%22%2C%22%22%2C1587967916%2C%22https%3A%2F%2Fwww.douban.com%2Fpeople%2F211922653%2F%22%5D; _pk_ses.100001.2fad=*; ap_v=0,6.0; __utma=30149280.2126318535.1587721127.1587958527.1587967956.8; __utmt=1; __utmb=30149280.4.10.1587967956; _pk_id.100001.2fad=192f2e02b4708ce9.1587867222.2.1587968004.1587868159.; login_start_time=1587968204395',
    }
    s.post(url=post_url, data=data, headers=headers)


    # 2、抓取具体数据
    get_url = 'https://www.douban.com/people/211922653/'
    get_headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36'}
    # get_headers中不可以存在Cookie这个字段,如果存在则找get_headers中的Cookie，反之会由 s 自动提交
    html = s.get(url=get_url, headers=get_headers).text
    print(html)

login()









