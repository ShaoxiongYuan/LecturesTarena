import csv
import time

from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.cache import cache_page


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



























