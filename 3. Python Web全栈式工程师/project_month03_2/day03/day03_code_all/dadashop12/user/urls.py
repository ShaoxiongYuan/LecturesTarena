from django.conf.urls import url
from . import views

urlpatterns = [
    #http://127.0.0.1:8000/v1/users
    url(r'^$', views.users),
    url(r'^/activation$', views.active_view),
    #/v1/users/<username>/address
    url(r'^/(?P<username>\w+)/address$', views.AddressView.as_view())

]