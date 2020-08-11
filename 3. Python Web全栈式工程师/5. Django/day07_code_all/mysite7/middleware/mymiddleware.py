from django.http import HttpResponse
from django.utils.deprecation import MiddlewareMixin

import re


class MyMiddleware(MiddlewareMixin):

    def process_request(self, request):

        print('---process request in ')

    def process_view(self, request, callback, callback_args, callback_kwargs):

        print('---process view in')


    def process_response(self, request, response):

        print('---process res in')
        return response


class MyMiddleware2(MiddlewareMixin):

    def process_request(self, request):

        print('---process2 request in ')

    def process_view(self, request, callback, callback_args, callback_kwargs):

        print('---process2 view in')


    def process_response(self, request, response):

        print('---process2 res in')
        return response


class VisitLimit(MiddlewareMixin):

    visit_times = {}


    def process_request(self, request):

        ip_address = request.META['REMOTE_ADDR']
        # /test_mw  /test开头的请求都要计数
        if not re.match(r'^/test', request.path_info):
            return

        times = self.visit_times.get(ip_address, 0)
        print('IP %s 访问了 %s 次'%(ip_address, times))
        if times >= 5:
            return HttpResponse('---对不起,访问次数已达上限')

        self.visit_times[ip_address] = times + 1















