"""
使用selenium登录豆瓣网 - iframe
"""
from selenium import webdriver

driver = webdriver.Chrome()
driver.maximize_window()
driver.get(url='https://www.douban.com/')

# 1、先切换到iframe子页面
node = driver.find_element_by_xpath('//*[@id="anony-reg-new"]/div/div[1]/iframe')
driver.switch_to.frame(node)

# 2、密码登录、用户名、密码、登录豆瓣按钮
driver.find_element_by_xpath('/html/body/div[1]/div[1]/ul[1]/li[2]').click()
driver.find_element_by_id('username').send_keys('15110225726')
driver.find_element_by_id('password').send_keys('zhanshen001')

# 方法一 ：通过 link_text() 找到登录节点
driver.find_element_by_link_text('登录豆瓣').click()

# 方法二 ：通过 class 属性值找到登录节点
# 当属性值中存在 空格 时,我们要使用 . 去代替空格
# driver.find_element_by_class_name('btn.btn-account').click()















