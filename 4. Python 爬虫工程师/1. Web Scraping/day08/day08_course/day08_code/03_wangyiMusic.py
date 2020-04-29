"""
selenium抓取网易音乐排行榜
思路：
    1、打开主页,切换iframe
    2、提取数据
    3、完成后关闭浏览器
"""
from selenium import webdriver
import pymongo

class WangYiSpider:
    def __init__(self):
        # 无界面模式,打开网易云音乐排行榜
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.get('https://music.163.com/#/discover/toplist')
        # mongodb相关变量
        self.conn = pymongo.MongoClient('localhost',27017)
        self.db = self.conn['musicdb']
        self.myset = self.db['musicset']

    def parse_html(self):
        # 切换到iframe子页面
        self.driver.switch_to.frame('contentFrame')
        # 基准xpath：提取所有歌曲的 tr 节点对象列表
        tr_list = self.driver.find_elements_by_xpath('//tbody/tr')
        for tr in tr_list:
            item = {}
            # 歌曲排名、歌曲名称、歌曲时长、歌手
            item['music_number'] = tr.find_element_by_xpath('.//span[@class="num"]').text
            item['music_name'] = tr.find_element_by_xpath('.//span[@class="txt"]/a/b').get_attribute('title').replace('\xa0',' ')
            item['music_time'] = tr.find_element_by_xpath('.//span[@class="u-dur "]').text
            item['music_star'] = tr.find_element_by_xpath('.//div[@class="text"]').get_attribute('title')
            self.myset.insert_one(item)
            print(item)

        # 提取完成后关闭浏览器
        self.driver.quit()

    def run(self):
        """程序入口函数"""
        self.parse_html()

if __name__ == '__main__':
    spider = WangYiSpider()
    spider.run()