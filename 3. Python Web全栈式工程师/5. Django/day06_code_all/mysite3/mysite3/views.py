from django.http import HttpResponse
from django.shortcuts import render


def test_static(request):

    return render(request, 'test_static.html')


def test_set_cookies(request):

    resp = HttpResponse('test set cookies ok')

    #resp.set_cookie('name', 'guoxiaonao', 600)

    resp.delete_cookie('name')


    return resp


def test_get_cookies(request):

    value = request.COOKIES.get('name', 'no data')

    return HttpResponse('Cookies name value is %s'%(value))


def set_session(request):

    request.session['uname'] = 'wangweichao'

    return HttpResponse('set session is ok')

def get_session(request):

    value = request.session.get('uname', 'no data')
    return HttpResponse('session uname is %s'%(value))
























