"""
打开浏览器,输入百度,停留在页面
"""
# 导入webdriver接口
from selenium import webdriver

# 1、打开浏览器 - 创建浏览器对象
driver = webdriver.Firefox()
# 2、输入URL地址
driver.get('http://www.baidu.com/')
# 3、获取屏幕截图
driver.save_screenshot('baidu.png')

print(driver.page_source.find('pn-next disabled'))







