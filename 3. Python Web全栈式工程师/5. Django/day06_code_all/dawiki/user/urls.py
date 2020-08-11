from django.urls import path
from . import views

urlpatterns = [
    #http://127.0.0.1:8000/user/reg
    path('reg', views.reg_view),
    #http://127.0.0.1:8000/user/login
    path('login', views.login_view),
    #http://127.0.0.1:8000/user/logout
    path('logout', views.logout_view)

]