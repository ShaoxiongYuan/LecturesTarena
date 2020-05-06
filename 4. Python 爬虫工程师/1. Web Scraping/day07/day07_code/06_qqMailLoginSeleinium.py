"""
selenium登录qq邮箱
"""
from selenium import webdriver

driver = webdriver.Chrome()
driver.get(url='https://mail.qq.com/')

# 1、切换到frame
driver.switch_to.frame('login_frame')

# 2、找对应节点登录
driver.find_element_by_id('u').send_keys('2621470058')
driver.find_element_by_id('p').send_keys('zhanshen001')
driver.find_element_by_class_name('btn').click()


















