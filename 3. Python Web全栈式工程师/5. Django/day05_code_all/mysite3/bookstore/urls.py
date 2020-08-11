from django.urls import path

from . import views

urlpatterns = [

    #http://127.0.0.1:8000/bookstore/all_book
    path('all_book', views.all_book),
    #http://127.0.0.1:8000/bookstore/update_book/id
    path('update_book/<int:book_id>', views.update_book),
    path('delete_book', views.delete_book)
]