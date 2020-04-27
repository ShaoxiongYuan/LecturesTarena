"""
打开百度，输入赵丽颖，点击百度一下按钮
"""
from selenium import webdriver

# 1、打开浏览器,并最大化窗口
driver = webdriver.Chrome()
driver.maximize_window()
# 2、输入地址
driver.get('http://www.baidu.com/')
# 3、找到搜索框节点,并发送赵丽颖文本
driver.find_element_by_xpath('//*[@id="kw"]').send_keys('赵丽颖')
# 4、找到百度一下按钮节点,并点击
driver.find_element_by_xpath('//*[@id="su"]').click()
# 5、关闭浏览器
driver.quit()













