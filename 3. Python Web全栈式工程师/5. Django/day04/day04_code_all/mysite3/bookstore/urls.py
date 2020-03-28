from django.conf.urls import url
from . import views

urlpatterns = [
    #http://127.0.0.1:8000/bookstore/all_book
    url(r'^all_book$', views.all_book, name='all_book'),
    url(r'^update_book/(\d+)$', views.update_book),
    url(r'^delete_book/(\d+)$', views.delete_book)

]