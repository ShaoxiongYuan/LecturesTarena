"""
selenium实现抓取有道翻译结果
思路：
    1、找到输入翻译单词节点,发送文字
    2、休眠一定时间,等待网站给出响应-翻译结果
    3、找到翻译结果节点,获取文本内容
"""
from selenium import webdriver
import time

class YdSpider:
    def __init__(self):
        self.url = 'http://fanyi.youdao.com/'
        # 设置无界面模式
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=self.options)
        # 打开有道翻译官网
        self.driver.get(self.url)

    def parse_html(self, word):
        # 发送翻译单词
        self.driver.find_element_by_id('inputOriginal').send_keys(word)
        time.sleep(1)
        # 获取翻译结果
        result = self.driver.find_element_by_xpath('//*[@id="transTarget"]/p/span').text

        return result

    def run(self):
        word = input('请输入要翻译的单词:')
        print(self.parse_html(word))
        self.driver.quit()

if __name__ == '__main__':
    spider = YdSpider()
    spider.run()