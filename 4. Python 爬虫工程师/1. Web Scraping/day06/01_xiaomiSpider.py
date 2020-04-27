import requests
from fake_useragent import UserAgent
import json
from threading import Thread,Lock
from queue import Queue
import time
import random

class XiaomiSpider:
    def __init__(self):
        self.url = 'http://app.mi.com/categotyAllListApi?page={}&categoryId=2&pageSize=30'
        self.headers = {'User-Agent': UserAgent().random}
        self.q = Queue()
        self.lock = Lock()
        # 计数
        self.i = 0
        # 存放所有字典的大列表
        self.all_app_list = []

    # URL入队列
    def url_in(self):
        for page in range(67):
            url = self.url.format(page)
            self.q.put(url)

    # 线程事件函数
    def parse_html(self):
        while True:
            # 加锁 - 防止出现死锁(self.q中剩余1个地址,但是被多个线程判断的情况)
            self.lock.acquire()
            if not self.q.empty():
                url = self.q.get()
                # 获取地址成功后马上释放锁,给其他线程机会,安全前提下提升效率
                self.lock.release()
                # 请求 + 解析  html: {'count':2000,'data':[{},{},{}]}
                try:
                    res = requests.get(url=url, headers=self.headers)
                    html = json.loads(res.text)
                    for one_app in html['data']:
                        item = {}
                        item['app_name'] = one_app['displayName']
                        item['app_type'] = one_app['level1CategoryName']
                        item['app_link'] = one_app['packageName']
                        print(item)

                        # 加锁+释放锁
                        self.lock.acquire()
                        self.all_app_list.append(item)
                        self.i += 1
                        self.lock.release()
                    # 简单控制一下数据抓取频率,因为我们没有代理IP,容易被封掉IP
                    time.sleep(random.uniform(0,1))
                except Exception as e:
                    print(e)
            else:
                # 如果队列为空了,上面已经上锁,所以此处释放锁
                self.lock.release()
                break

    # 入口函数
    def run(self):
        # 1.先让URL地址入队列
        self.url_in()
        # 2.多线程,开始执行
        t_list = []
        for i in range(2):
            t = Thread(target=self.parse_html)
            t_list.append(t)
            t.start()

        for j in t_list:
            j.join()

        print('数量:', self.i)
        with open('xiaomi.json','w',encoding='utf-8') as f:
            json.dump(self.all_app_list,f,ensure_ascii=False)

if __name__ == '__main__':
    start_time = time.time()
    spider = XiaomiSpider()
    spider.run()
    end_time = time.time()
    print('执行时间:%.2f' % (end_time - start_time))