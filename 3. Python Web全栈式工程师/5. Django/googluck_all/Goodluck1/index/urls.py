from django.urls import path
from . import views
urlpatterns = [
    path('get_page/', views.get_page),
    path('get_prize/', views.get_prize)
]
