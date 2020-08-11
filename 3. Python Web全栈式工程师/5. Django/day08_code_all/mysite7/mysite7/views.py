import csv
import os
import time

from django.core.paginator import Paginator
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.cache import cache_page

from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from test_upload.models import Content


@cache_page(60)
def test_cache(request):
    #time.sleep(3)
    t1 = time.time()
    return HttpResponse('t1 is %s'%(t1))
    #return render(request, 'test_cache.html', locals())



def test_mw(request):

    print('---view in')

    return HttpResponse('--test mw ok')

def test_csrf(request):

    if request.method == 'GET':
        return render(request, 'test_csrf.html')
    elif request.method == 'POST':
        return HttpResponse('--post ok')


def test_csv(request):


    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment;filename="book.csv"'

    all_book = [{'id':1, 'title':'python1'}, {'id':2, 'title':'python2'}]

    #生成csv写对象
    writer = csv.writer(response)
    writer.writerow(['id', 'title'])
    for book in all_book:
        writer.writerow([book['id'], book['title']])

    return response


def test_page(request):
    # /test_page?page=1 方案1
    # /test_page/1   方案2
    bks = ['a', 'b', 'c', 'd', 'e']
    paginator = Paginator(bks, 2)

    cur_page = request.GET.get('page', 1)
    page = paginator.page(cur_page)

    return render(request, 'test_page.html', locals())

@csrf_exempt
def test_upload(request):

    if request.method == 'GET':
        return render(request, 'test_upload.html')

    elif request.method == 'POST':
        #处理上传数据
        #请求头 Content-Type:multipart/form-data; boundary=---------------------------137840814639045439627491309
        # 方案1
        # a_file = request.FILES['myfile']
        # print("上传文件名是:", a_file.name)
        # filename = os.path.join(settings.MEDIA_ROOT, a_file.name)
        # with open(filename, 'wb') as f:
        #     data = a_file.file.read()
        #     f.write(data)

        #方案2
        title = request.POST['title']
        a_file = request.FILES['myfile']

        Content.objects.create(desc=title, myfile=a_file)


        return HttpResponse("接收文件:" + a_file.name + "成功")








































