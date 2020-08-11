from django.contrib import admin
from .models import Book, Author
# Register your models here.

class BookManager(admin.ModelAdmin):

    list_display = ['id', 'title', 'pub', 'price']
    #该值必须在list_display中
    list_display_links = ['title']
    list_filter = ['pub']
    search_fields = ['title']
    # 1,该值必须在list_display中; 2,不能在list_display_links里
    list_editable = ['price']



class AuthorManager(admin.ModelAdmin):

    list_display = ['id', 'name', 'age']




admin.site.register(Book, BookManager)
admin.site.register(Author, AuthorManager)



