"""
京东商品信息抓取
"""
from selenium import webdriver
import time
import pymongo

class JdSpider:
    def __init__(self):
        # 设置无头模式
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.get(url='https://www.jd.com/')
        # 3个对象
        self.conn = pymongo.MongoClient('localhost', 27017)
        self.db = self.conn['jddb']
        self.myset = self.db['jdset']

    def get_html(self):
        """输入爬虫书,点击搜索按钮,到商品页"""
        self.driver.find_element_by_xpath('//*[@id="key"]').send_keys('爬虫书')
        self.driver.find_element_by_xpath('//*[@id="search"]/div/div[2]/button').click()
        # 给页面元素加载预留时间
        time.sleep(1)

    def parse_html(self):
        """拉动滚动条到底部,提取数据"""
        # 执行JS脚本,将滚动条拉动
        self.driver.execute_script(
            'window.scrollTo(0,document.body.scrollHeight)'
        )
        time.sleep(2)
        # 基准xpath,匹配每个商品的li节点对象列表
        li_list = self.driver.find_elements_by_xpath('//*[@id="J_goodsList"]/ul/li')
        for li in li_list:
            item = {}
            item['name'] = li.find_element_by_xpath('.//div[@class="p-name"]/a/em').text
            item['price'] = li.find_element_by_xpath('.//div[@class="p-price"]').text
            item['commit'] = li.find_element_by_xpath('.//div[@class="p-commit"]/strong').text
            item['shop'] = li.find_element_by_xpath('.//div[@class="p-shopnum"]').text
            print(item)
            self.myset.insert_one(item)

    def run(self):
        self.get_html()
        while True:
            self.parse_html()
            # 判断是否为最后一页
            if self.driver.page_source.find('pn-next disabled') == -1:
                self.driver.find_element_by_xpath('//*[@id="J_bottomPage"]/span[1]/a[9]').click()
                time.sleep(1)
            else:
                self.driver.quit()
                break

if __name__ == '__main__':
    spider = JdSpider()
    spider.run()











