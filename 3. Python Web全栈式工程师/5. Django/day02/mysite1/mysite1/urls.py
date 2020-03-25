"""mysite1 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from . import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    #http://127.0.0.1:8000/page1
    url(r'^page1$', views.page1_view),
    #http://127.0.0.1:8000/page2
    url(r'^page2$', views.page2_view),
    #http://127.0.0.1:8000/
    url(r'^$', views.index_view),
    #http://127.0.0.1:8000/page3~100
    #普通分组() -> 位置传参的方式 传递给视图函数
    url(r'^page(\d+)', views.pagen_view),
    #http://127.0.0.1:8000/100/add/200
    url(r'^(\d+)/(\w+)/(\d+)', views.cal_view),
    #http://127.0.0.1:8000/person/guoxiaonao/18
    url(r'^person/(?P<name>\w+)/(?P<age>\d{1,2})',views.person_view),

    url(r'^birthday/(?P<y>\d{4})/(?P<m>\d{1,2})/(?P<d>\d{1,2})', views.birthday_view),
    url(r'^birthday/(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{4})', views.birthday_view),
    #http://127.0.0.1:8000/test_get
    url(r'^test_get$',views.test_get),
    #http://127.0.0.1:8000/test_post
    url(r'^test_post$', views.test_post)
]
