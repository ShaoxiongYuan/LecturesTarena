"""
selenium切换句柄,抓取民政部最新行政区划代码
思路：
    1、找到最新月份链接
    2、点击之后,切换句柄
    3、正常数据提取
"""
from selenium import webdriver
import time
import redis
from hashlib import md5
import sys

class MzbSpider:
    def __init__(self):
        self.url = 'http://www.mca.gov.cn/article/sj/xzqh/2020/'
        self.driver = webdriver.Chrome()
        # 只有get()方法,不需要time.sleep()等待,因为get()方法会等待页面所有元素加载完成后才会继续执行
        self.driver.get(url=self.url)
        # redis实现增量爬虫
        self.r = redis.Redis(host='127.0.0.1', port=6379, db=0)

    def md5_url(self, url):
        """对URL地址进行md5加密"""
        s = md5()
        s.update(url.encode())

        return s.hexdigest()

    def parse_html(self):
        # 1、找最新月份节点,并点击
        a_node = self.driver.find_element_by_partial_link_text('中华人民共和国县以上行政区划代码')
        href = a_node.get_attribute('href')
        finger = self.md5_url(href)
        # 判断是否需要抓取
        if self.r.sadd('mzb:spider', finger) == 1:
            a_node.click()
            time.sleep(1)
            # 2、切换句柄: all是列表
            all = self.driver.window_handles
            self.driver.switch_to.window(all[1])
            # 3、提取数据
            tr_list = self.driver.find_elements_by_xpath('//tr[@height="19"]')
            for tr in tr_list:
                item = {}
                item['name'] = tr.find_element_by_xpath('./td[3]').text.strip()
                item['code'] = tr.find_element_by_xpath('./td[2]').text.strip()
                print(item)
        else:
            self.driver.quit()
            sys.exit('完成')

    def run(self):
        self.parse_html()

if __name__ == '__main__':
    spider = MzbSpider()
    spider.run()













