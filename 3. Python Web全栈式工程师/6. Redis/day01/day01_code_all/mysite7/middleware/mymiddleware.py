from django.http import HttpResponse
from django.utils.deprecation import MiddlewareMixin


class MyMW(MiddlewareMixin):

    def process_request(self, request):

        print('MyMW process_request do ---')

    def process_view(self, request, callback, callback_args, callback_kwargs):

        print('MyMW process_view do ---')

    def process_response(self, request, response):
        #必须返回 HttpResponse对象
        print('MyMW process_response do ---')
        return response


class MyMW2(MiddlewareMixin):

    def process_request(self, request):

        print('MyMW2 process_request do ---')

    def process_view(self, request, callback, callback_args, callback_kwargs):

        print('MyMW2 process_view do ---')

    def process_response(self, request, response):
        #必须返回 HttpResponse对象
        print('MyMW2 process_response do ---')
        return response

class VisitLimit(MiddlewareMixin):

    visit_times = {}
    def process_request(self, request):

        ip_address = request.META['REMOTE_ADDR']
        path = request.path_info
        if path != '/test_mw':
            return
        #if not re.match('^/test', request.path_info):
        #    return
        times = self.visit_times.get(ip_address, 0)
        self.visit_times[ip_address] = times + 1
        if times < 5:
            return
        return HttpResponse('您已经访问过 %s次， 您被禁止了'%(times))




















