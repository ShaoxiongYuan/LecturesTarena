import hashlib
import json
import time

from django.http import JsonResponse
from django.shortcuts import render
from .models import UserProfile
from django.conf import settings
from dtoken.views import make_token


#10100 - 10199

# Create your views here.
def users(request):

    if request.method != 'POST':
        return JsonResponse({'code':10100, 'error':'Please use POST !'})

    json_str = request.body
    data = json.loads(json_str)
    username = data['uname']
    email = data['email']
    password = data['password']
    phone = data['phone']
    #TODO 参数判断

    #检查用户名是否已经存在
    old_users = UserProfile.objects.filter(username=username)
    if old_users:
        result = {'code':10101, 'error': 'Your username is already existed!'}
        return JsonResponse(result)

    m = hashlib.md5()
    m.update(password.encode())

    #创建用户
    try:
        user = UserProfile.objects.create(username=username,password=m.hexdigest(),email=email,phone=phone)
    except Exception as e:
        print('---create error is ---')
        print(e)
        return JsonResponse({'code':10102, 'error':'Your username is already existed~~'})

    #生成令牌
    token = make_token(username)

    #容错性
    try:
        #激活邮件的功能
        #1,给用户邮箱中 发邮件，邮件内容 - ‘欢迎来到达达商城，点击如下链接，可完成注册’
        #2,邮件中的链接有效期为3天
        #3,code的隐私性
        # send_mail(html_message='')
        verify_url = 'http://127.0.0.1:7000/dadashop/templates/active.html?code=xxx'
    except Exception as e:
        pass

    return JsonResponse({'code':200,'username':username, 'data':{'token':token.decode()},'carts_count':0})
















