from django.http import HttpResponse
from django.shortcuts import render


def test_static(request):

    return render(request, 'test_static.html')


def test_set_cookies(request):

    resp = HttpResponse('test set cookies ok')

    resp.set_cookie('uuname', 'guoxiaonao', 600)

    return resp








