# -*- coding: utf-8 -*-

# Define here the models for your spider middleware
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/spider-middleware.html

from scrapy import signals


class MiddleSpiderMiddleware(object):
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, dict or Item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Response, dict
        # or Item objects.
        pass

    def process_start_requests(self, start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)


class MiddleDownloaderMiddleware(object):
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the downloader middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):
        # Called for each request that goes through the downloader
        # middleware.

        # Must either:
        # - return None: continue processing this request
        # - or return a Response object
        # - or return a Request object
        # - or raise IgnoreRequest: process_exception() methods of
        #   installed downloader middleware will be called
        return None

    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.

        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)

# 中间件1 - 为每个请求包装随机的User-Agent
from fake_useragent import UserAgent

class MiddleRandomUADownloaderMiddleware(object):
    def process_request(self, request, spider):
        """给请求对象赋值一个随机的User-Agent"""
        # 请求对象request的属性,headers为一个字典, User-Agent只是字典中一个键值对
        agent = UserAgent().random
        request.headers['User-Agent'] = agent
        print(agent)

# 中间件2 - 随机代理中间件
from .proxypool import proxy_list
import random

class MiddleRandomProxyDownloaderMiddleware(object):
    def process_request(self, request, spider):
        proxy = random.choice(proxy_list)
        # 利用meta属性添加代理
        request.meta['proxy'] = proxy
        print(proxy)

    def process_exception(self, request, exception, spider):
        """当捕捉到异常后（代表代理IP不能用）：再次走中间件才对"""
        return request

# 中间件3 - 添加cookie
class MiddleCookieDownloaderMiddleware(object):
    def process_request(self, request, spider):
        cookies = self.get_cookies()
        request.cookies = cookies
        print(cookies)

    def get_cookies(self):
        """把cookie处理成字典"""
        cook_str = 'BIDUPSID=46D0471B72D849FC7EDF21BA4702F83C; PSTM=1587698693; BD_UPN=12314753; sug=0; sugstore=1; ORIGIN=0; bdime=0; BAIDUID=46D0471B72D849FCE9A270A451DF87D1:SL=0:NR=10:FG=1; COOKIE_SESSION=75897_1_9_9_44_15_0_4_9_4_0_0_0_0_3_1_1588149939_1588225834_1588225833%7C9%230_1_1588225833%7C1; BD_HOME=1; H_PS_PSSID=1463_31326_21107_31427_31341_31464_31228_30824_31164_31472; delPer=0; BD_CK_SAM=1; PSINO=2; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598'
        cookies = {}
        for kv in cook_str.split('; '):
            key = kv.split('=')[0]
            value = kv.split('=')[1]
            cookies[key] = value

        return cookies








