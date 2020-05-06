"""
腾讯招聘具体职位信息抓取
思路：
    1、一级页面中：每个职位的POSTID（用来拼接职位详情页的URL地址）
    2、二级页面中：提取职位相信信息
    3、再使用redis实现增量爬虫
"""
import requests
import time
from threading import Thread,Lock
from queue import Queue
from fake_useragent import UserAgent
from urllib import parse
import redis
from hashlib import md5

class TencentSpider:
    def __init__(self):
        self.one_url = 'https://careers.tencent.com/tencentcareer/api/post/Query?timestamp=1563912271089&countryId=&cityId=&bgIds=&productId=&categoryId=&parentCategoryId=&attrId=&keyword={}&pageIndex={}&pageSize=10&language=zh-cn&area=cn'
        self.two_url = 'https://careers.tencent.com/tencentcareer/api/post/ByPostId?timestamp=1563912374645&postId={}&language=zh-cn'
        # 2个队列、2把锁
        self.one_q = Queue()
        self.two_q = Queue()
        self.lock1 = Lock()
        self.lock2 = Lock()
        # 计数
        self.count = 0
        # 连接redis
        self.r = redis.Redis(host='localhost', port=6379, db=0)

    def get_html(self, url):
        """功能函数: 获取响应内容"""
        headers = { 'User-Agent' : UserAgent().random }
        html = requests.get(url=url, headers=headers).json()

        return html

    def md5_url(self, url):
        """功能函数：生成请求指纹"""
        s = md5()
        s.update(url.encode())

        return s.hexdigest()

    def url_in(self):
        """让一级页面URL地址入队列: self.one_q"""
        keyword = input('请输入职位类别:')
        keyword = parse.quote(keyword)
        # 计算总页数
        total = self.get_total(keyword)
        for index in range(1, total+1):
            one_url = self.one_url.format(keyword, index)
            # URL地址入一级队列
            self.one_q.put(one_url)

    def get_total(self, keyword):
        """获取keyword类别下的职位总页数"""
        url = self.one_url.format(keyword, 1)
        html = self.get_html(url=url)
        cou = html['Data']['Count']
        total = cou//10 if cou%10==0 else cou//10 + 1

        return total

    def parse_one_page(self):
        """一级页面线程事件函数,提取数据: postid,并拼接详情页链接交给二级队列"""
        while True:
            # 加锁
            self.lock1.acquire()
            if not self.one_q.empty():
                one_url = self.one_q.get()
                # 释放锁
                self.lock1.release()
                one_html = self.get_html(url=one_url)
                # 在one_html中提取10个职位的postid
                for one_job in one_html['Data']['Posts']:
                    post_id = one_job['PostId']
                    # 拼接每个职位详情页链接,交给二级队列
                    two_url = self.two_url.format(post_id)
                    self.two_q.put(two_url)
            else:
                self.lock1.release()
                break

    def parse_two_page(self):
        """二级页面线程事件函数: 获取每个职位的具体信息"""
        while True:
            try:
                two_url = self.two_q.get(block=True, timeout=2)
                # 生成指纹,来判断是否需要继续抓取此职位
                finger = self.md5_url(url=two_url)
                if self.r.sadd('tencent:spider', finger) == 1:
                    two_html = self.get_html(url=two_url)
                    item = {}
                    item['job_name'] = two_html['Data']['RecruitPostName'].strip()
                    item['job_address'] = two_html['Data']['LocationName'].strip()
                    item['job_type'] = two_html['Data']['CategoryName'].strip()
                    item['job_time'] = two_html['Data']['LastUpdateTime'].strip()
                    item['job_duty'] = two_html['Data']['Responsibility'].strip()
                    item['job_require'] = two_html['Data']['Requirement'].strip()
                    print(item)
                    # 加锁、释放锁
                    self.lock2.acquire()
                    self.count += 1
                    self.lock2.release()
                else:
                    pass
            except Exception as e:
                print(e)
                break

    def run(self):
        """程序入口函数"""
        self.url_in()
        t1_list = []
        t2_list = []
        for i in range(3):
            t1 = Thread(target=self.parse_one_page)
            t1_list.append(t1)
            t1.start()

        for j in range(3):
            t2 = Thread(target=self.parse_two_page)
            t2_list.append(t2)
            t2.start()

        for t in t1_list:
            t.join()

        for t in t2_list:
            t.join()

        print('job number:', self.count)

if __name__ == '__main__':
    start_time = time.time()
    spider = TencentSpider()
    spider.run()
    end_time = time.time()
    print('time:%.2f' % (end_time-start_time))










