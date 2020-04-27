"""
使用selenium+Chrome抓取猫眼电影top100
知识点：
    1、通过 find_element_by_link_text('下一页') 来锁定下一页按钮节点
    2、最后1页时,下一页按钮消失, find_element_by_xxxx() 如果找不到节点会抛出异常
"""
from selenium import webdriver

driver = webdriver.Chrome()
driver.get('https://maoyan.com/board/4?offset=0')

def parse_html():
    # 1、基准xpath,来匹配每个电影信息的dd节点对象列表 - 必须使用find_elements
    dd_list = driver.find_elements_by_xpath('//*[@id="app"]/div/div/div[1]/dl/dd')
    for dd in dd_list:
        # text属性: 获取当前节点以及子节点和后代节点的文本内容
        film_list = dd.text.split('\n')
        item = {}
        item['rank'] = film_list[0]
        item['name'] = film_list[1]
        item['star'] = film_list[2]
        item['time'] = film_list[3]
        item['score'] = film_list[4]
        print(item)

# 抓取所有页
while True:
    parse_html()
    try:
        # selenium当找不到某个节点时,会抛出异常
        driver.find_element_by_link_text('下一页').click()
    except Exception as e:
        driver.quit()
        break















