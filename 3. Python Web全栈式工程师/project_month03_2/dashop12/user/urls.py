# -*- coding:utf-8 -*-
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.Users.as_view()),
    url(r'^/activation$', views.ActiveView.as_view()),
    # 用户登陆状态下修改密码
    url(r'^/(?P<username>[\w]{1,20})/password$', views.ModifyPasswordView.as_view()),

    # 用户忘记密码修改
    url(r'^/password/sms$', views.SendSmsCodeView.as_view()),
    url(r'^/password/verification$', views.VerifyCodeView.as_view()),
    url(r'^/password/new$', views.ModifyPwdView.as_view()),

    # 地址的增查
    url(r'^/(?P<username>[\w]{1,20})/address$', views.AddressView.as_view()),
    # 改 删
    url(r'^/(?P<username>[\w]{1,20})/address/(?P<id>[\d]{1,5})$', views.AddressView.as_view()),
    # 对用户默认地址的设置
    url(r'^/(?P<username>[\w]{1,20})/address/default$', views.DefaultAddressView.as_view()),

    # 用于重定向微博授权页面
    url(r'^/weibo/authorization$', views.OAuthWeiboUrlView.as_view()),
    # 用于获取到code之后向微博服务器发送请求
    url(r'^/weibo/users', views.OAuthWeiboView.as_view()),

    # 发送短信验证码接口
    #url(r'^/sms/code', views.SmScodeView.as_view())
]

