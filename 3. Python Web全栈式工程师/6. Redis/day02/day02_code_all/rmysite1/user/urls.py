from django.conf.urls import url

from . import views

urlpatterns = [
    #http://127.0.0.1:8000/user/detail/1
    url(r'^detail/(\d+)$', views.user_detail),
    #http://127.0.0.1:8000/user/update?user_id=1&nickname=gxn
    url(r'^update$', views.user_update)

]