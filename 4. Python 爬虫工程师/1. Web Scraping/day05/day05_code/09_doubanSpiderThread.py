"""
使用多线程抓取豆瓣电影 - 剧情
思路：
    1、队列中放入URL地址
    2、线程事件函数,从队列中获取地址,请求,解析,保存
    3、创建多个线程去执行
注意：
    1、注意线程锁问题（只要多个线程操作共享资源了,一定要加锁和释放锁）
"""
import requests
from threading import Thread,Lock
from queue import Queue
import time
from fake_useragent import UserAgent
import json

class DoubanSpider:
    def __init__(self):
        self.url = 'https://movie.douban.com/j/chart/top_list?type=11&interval_id=100%3A90&action=&start={}&limit=20'
        # 创建队列
        self.q = Queue()
        # 创建锁
        self.lock = Lock()
        # 打开json文件
        self.f = open('douban.json', 'w', encoding='utf-8')
        self.all_list = []

    def url_in(self):
        """将所有待抓取的URL地址入队列"""
        for start in range(0, 692, 20):
            page_url = self.url.format(start)
            self.q.put(page_url)

    def get_html(self, url):
        """请求功能函数"""
        headers = { 'User-Agent':UserAgent().random }
        html = requests.get(url=url, headers=headers).json()

        return html

    def parse_html(self):
        """线程事件函数: 先从队列中get()地址,请求,解析,保存"""
        while True:
            # 加锁：因为操作self.q
            self.lock.acquire()
            if not self.q.empty():
                url = self.q.get()
                # 获取到地址后,马上释放锁,给其他线程腾出资源
                self.lock.release()
                html = self.get_html(url=url)
                for one_film in html:
                    item = {}
                    item['rank'] = one_film['rank']
                    item['name'] = one_film['title']
                    item['time'] = one_film['release_date']
                    item['score'] = one_film['score']
                    print(item)
                    # 操作全局变量,加锁和释放锁
                    self.lock.acquire()
                    self.all_list.append(item)
                    self.lock.release()
            else:
                # 当队列为空时,保证释放锁,其他线程还会判断
                self.lock.release()
                break

    def run(self):
        """程序入口函数"""
        self.url_in()
        t_list = []
        for i in range(3):
            t = Thread(target=self.parse_html)
            t_list.append(t)
            t.start()

        for t in t_list:
            t.join()

        # 把数据保存到json文件中
        json.dump(self.all_list, self.f, ensure_ascii=False)
        self.f.close()

if __name__ == '__main__':
    start_time = time.time()
    spider = DoubanSpider()
    spider.run()
    end_time = time.time()
    print('time:%.2f' % (end_time-start_time))























