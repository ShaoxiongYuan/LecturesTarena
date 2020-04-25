"""
收费代理：
    建立私密代理的代理IP池
思路：
    1、获取到私密代理
    2、依次对每个代理IP进行测试,能用的保存到文件中
"""
import requests

class ProxyPool:
    def __init__(self):
        self.url = 'http://dps.kdlapi.com/api/getdps/?orderid=958685347398257&num=20&pt=1&sep=1'
        self.headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36'}
        # 打开文件,用来存放可用的代理IP
        self.f = open('proxy.txt', 'w')

    def get_html(self):
        html = requests.get(url=self.url, headers=self.headers).text
        proxy_list = html.split('\r\n')
        for proxy in proxy_list:
            # 依次测试每个代理IP是否可用
            if self.check_proxy(proxy):
                self.f.write(proxy + '\n')

    def check_proxy(self, proxy):
        """测试1个代理IP是否可用,可用返回True,否则返回False"""
        test_url = 'http://httpbin.org/get'
        proxies = {
            'http' : 'http://309435365:szayclhp@{}'.format(proxy),
            'https': 'https://309435365:szayclhp@{}'.format(proxy)
        }
        try:
            res = requests.get(url=test_url, proxies=proxies, headers=self.headers, timeout=2)
            if res.status_code == 200:
                print(proxy,'\033[31m可用\033[0m')
                return True
            else:
                print(proxy,'无效')
                return False
        except:
            print(proxy,'无效')
            return False

    def run(self):
        self.get_html()
        # 关闭文件
        self.f.close()

if __name__ == '__main__':
    spider = ProxyPool()
    spider.run()











