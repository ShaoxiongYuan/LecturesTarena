from django.conf.urls import url
from .views import *

urlpatterns = [
    url(r'^/(?P<username>[\w]{1,11})$',CartsView.as_view()),
]
