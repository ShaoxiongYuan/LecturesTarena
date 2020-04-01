from django.http import HttpResponse
from django.shortcuts import render


def test_static(request):
    return render(request, 'test_static.html')


def test_cookies(request):
    username = request.COOKIES.get('username', 'hahahahah')
    print('---test--cookies---')
    print(username)
    resp = HttpResponse('---哈哈哈---')
    resp.set_cookie('username', 'guoxiaonao', 300)
    # resp.set_cookie('user', 'guoxiaonao')
    return resp


def set_session(request):
    request.session['uname'] = 'guoxiaonao'
    return HttpResponse('---set session is ok---')


def get_session(request):
    name = request.session['uname']
    return HttpResponse('---get session values is %s' % (name))
