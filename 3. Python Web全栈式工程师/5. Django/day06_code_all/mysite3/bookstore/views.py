from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from .models import Book



# Create your views here.
def all_book(request):

    all_book = Book.objects.filter(is_active=True)
    return render(request, 'bookstore/all_book.html', locals())


def update_book(request, book_id):

    #return HttpResponse('这是%s的详情页'%(book_id))

    try:
        book = Book.objects.get(id=book_id, is_active=True)
    except Exception as e:
        print('--update book error %s'%(e))
        return HttpResponse('---book id error---')

    if request.method == 'GET':

        return render(request, 'bookstore/update_book.html', locals())

    elif request.method == 'POST':
        #修改书的数据
        price = request.POST['price']
        market_price = request.POST['market_price']

        #改
        book.price = price
        book.market_price = market_price
        #保存

        book.save()

        #302跳转
        return HttpResponseRedirect('/bookstore/all_book')


def delete_book(request):

    bid = request.GET.get('bid')
    if not bid:
        return HttpResponse('--bid is error')

    #查
    try:
        book = Book.objects.get(id=bid, is_active=True)
    except Exception as e:
        print('---no book %s'%(e))
        return HttpResponse('---book is error')

    book.is_active = False
    book.save()
    return HttpResponseRedirect('/bookstore/all_book')
















