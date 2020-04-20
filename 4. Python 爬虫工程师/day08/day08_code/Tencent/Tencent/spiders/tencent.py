# -*- coding: utf-8 -*-
import scrapy
from urllib import parse
import requests
import json
from ..items import TencentItem


class TencentSpider(scrapy.Spider):
    name = 'tencent'
    allowed_domains = ['careers.tencent.com']
    one_url = 'https://careers.tencent.com/tencentcareer/api/post/Query?timestamp=1566266592644&countryId=&cityId=&bgIds=&productId=&categoryId=&parentCategoryId=&attrId=&keyword={}&pageIndex={}&pageSize=10&language=zh-cn&area=cn'
    two_url = 'https://careers.tencent.com/tencentcareer/api/post/ByPostId?timestamp=1566266695175&postId={}&language=zh-cn'
    headers = {'User-Agent': 'Mozilla/5.0'}
    keyword = input('请输入职位类别:')
    keyword = parse.quote(keyword)

    # 重写start_requests()方法
    def start_requests(self):
        """把所有一级页面的URL地址交给调度器入队列"""
        total = self.get_total()
        for index in range(1, total + 1):
            url = self.one_url.format(self.keyword, index)
            yield scrapy.Request(url=url, callback=self.parse_one_page)

    def get_total(self):
        """获取word类别的总页数"""
        url = self.one_url.format(self.keyword, 1)
        html = requests.get(url=url, headers=self.headers).json()
        count = html['Data']['Count']
        total = count // 10 if count % 10 == 0 else count // 10 + 1

        return total

    def parse_one_page(self, response):
        """一级页面解析函数: 提取 postid"""
        html = json.loads(response.text)
        for one in html['Data']['Posts']:
            # 此处是不是有URL需要交给调度器去入队列了？有！创建item对象！！！
            item = TencentItem()
            item['post_id'] = one['PostId']
            item['job_url'] = self.two_url.format(item['post_id'])
            # 交给调度器入队列
            yield scrapy.Request(
                url=item['job_url'], meta={'item': item}, callback=self.detail_page)

    def detail_page(self, response):
        """二级页面解析函数: 提取具体职位信息"""
        item = response.meta['item']
        # 此处没有继续交给调度器的请求了,所以我不需要再重新创建item对象了
        html = json.loads(response.text)
        # 名称+类别+职责+要求+地址+时间
        item['job_name'] = html['Data']['RecruitPostName']
        item['job_type'] = html['Data']['CategoryName']
        item['job_duty'] = html['Data']['Responsibility']
        item['job_require'] = html['Data']['Requirement']
        item['job_address'] = html['Data']['LocationName']
        item['job_time'] = html['Data']['LastUpdateTime']

        # 1个item数据抓取完成,交给管道处理
        yield item
