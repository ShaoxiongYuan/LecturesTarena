from django.contrib import admin
from .models import Book, Author


# Register your models here.
class BookManager(admin.ModelAdmin):
    list_display = ['id', 'title', 'pub', 'price']
    list_display_links = ['title']
    list_filter = ['pub']
    search_fields = ['title']
    list_editable = ['price']


class AuthorManager(admin.ModelAdmin):
    list_display = ['id', 'name', 'age']


admin.site.register(Author, AuthorManager)
admin.site.register(Book, BookManager)
