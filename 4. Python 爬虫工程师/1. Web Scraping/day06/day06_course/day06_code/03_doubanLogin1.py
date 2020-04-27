"""
抓取我的豆瓣主页 - 需要登录才能访问的页面
思路：
    1、先登录成功,F12抓取到对应的Cookie
    2、利用requests.get()中的 headers 参数
"""
import requests

def login():
    url = 'https://www.douban.com/people/211922653/'
    headers = {
        'Cookie' : 'll="118097"; bid=RXol6gm9_z8; _vwo_uuid_v2=D86CB471D688DB8877B43F317334F6048|560f53c46b5e7a6f5f2ac7fb1463f837; __utmz=30149280.1587803624.4.3.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; douban-profile-remind=1; push_noty_num=0; push_doumail_num=0; __utmc=30149280; _pk_ses.100001.8cb4=*; __utma=30149280.2126318535.1587721127.1587952263.1587958527.7; __utmt=1; dbcl2="211922653:YSkcfWeKUfQ"; ck=M01W; ap_v=0,6.0; __gads=ID=be0aec5eaed19432:T=1587958562:S=ALNI_MYseEt0nj3M1Xd_uO3mAgBK7QV1UA; __utmv=30149280.21192; _pk_id.100001.8cb4=672d49df57f4c9e4.1587866822.3.1587958825.1587952242.; __utmb=30149280.5.10.1587958527',
        'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36',
    }
    html = requests.get(url=url, headers=headers).text
    # 最终在html中确认: 是否存在个人主页的一些数据
    print(html)

login()















