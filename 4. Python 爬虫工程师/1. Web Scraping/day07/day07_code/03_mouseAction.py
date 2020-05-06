"""
selenium操作鼠标：打开百度 - 鼠标移动到设置节点 - 找到高级搜索 - 点击
"""
from selenium import webdriver
# 导入鼠标事件类
from selenium.webdriver import ActionChains

driver = webdriver.Chrome()
driver.maximize_window()
driver.get(url='http://www.baidu.com/')
# 1、找到设置节点,并将鼠标移动到此节点
node = driver.find_element_by_xpath('//*[@id="u1"]/a[9]')
ActionChains(driver).move_to_element(to_element=node).perform()

# 2、找到高级搜索节点,并点击
driver.find_element_by_link_text('高级搜索').click()















