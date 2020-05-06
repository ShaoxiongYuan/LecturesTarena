"""
设置无界面模式（无头模式）
"""
from selenium import webdriver

# 创建功能对象并添加无界面参数
options = webdriver.FirefoxOptions()
options.add_argument('--headless')

# 此处利用 options 参数
driver = webdriver.Firefox(options=options)
driver.get(url='http://www.baidu.com/')

driver.save_screenshot('baidu.png')


















