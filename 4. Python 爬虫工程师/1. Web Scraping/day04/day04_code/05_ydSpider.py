"""
抓取有道翻译的翻译结果
思路：
    1、翻译的结果为动态加载,F12开始抓包
    2、页面中翻译单词,在XHR中找到具体的网络数据包
    3、分析数据包
       3.1) Request URL：POST的URL地址
       3.2) Request Headers：请求头
       3.3) Form Data：Form表单数据
    4、html = requests.post(url=url, data=data, headers=headers).text
"""
import requests
import time
import random
from hashlib import md5

class YdSpider:
    def __init__(self):
        # url为F12抓包抓到的URL地址（POST）
        self.url = 'http://fanyi.youdao.com/translate_o?smartresult=dict&smartresult=rule'
        self.headers = {
            # 网站反爬中,检查频率最高的：User-Agent、Cookie、Referer
            "Cookie": "OUTFOX_SEARCH_USER_ID=2036516332@10.108.160.105; JSESSIONID=aaa1nlyVJA4mU9Vzu7Qgx; OUTFOX_SEARCH_USER_ID_NCOO=2061535381.1587195; SESSION_FROM_COOKIE=unknown; ___rl__test__cookies=1587715524620",
            "Referer": "http://fanyi.youdao.com/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36",
        }

    def get_ts_salt_sign(self, word):
        """功能函数：获取到ts、salt、sign用来完善form表单数据"""
        ts = str(int(time.time()*1000))
        salt = ts + str(random.randint(0,9))
        # sign
        string = "fanyideskweb" + word + salt + "Nw(nmmbP%A-r6U3EUn]Aj"
        s = md5()
        s.update(string.encode())
        sign = s.hexdigest()

        return ts, salt, sign

    def get_result(self, word):
        # 获取ts,salt,sign
        ts, salt, sign = self.get_ts_salt_sign(word)
        data = {
            "i": word,
            "from": "AUTO",
            "to": "AUTO",
            "smartresult": "dict",
            "client": "fanyideskweb",
            "salt": salt,
            "sign": sign,
            "ts": ts,
            "bv": "d50cccdf80e72b2ce6d88dd76d3f5cd6",
            "doctype": "json",
            "version": "2.1",
            "keyfrom": "fanyi.web",
            "action": "FY_BY_CLICKBUTTION",
        }
        # .json(): 直接获取python数据类型,前提是网站返回的数据必须为json类型
        html = requests.post(url=self.url, data=data, headers=self.headers).json()
        result = html['translateResult'][0][0]['tgt']

        return result

    def run(self):
        word = input('请输入要翻译的单词:')
        print(self.get_result(word))

if __name__ == '__main__':
    spider = YdSpider()
    spider.run()

