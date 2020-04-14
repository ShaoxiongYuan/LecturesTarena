import base64
import hashlib
import json
import random
import time

from django.core.mail import send_mail
from django.http import JsonResponse
from django.shortcuts import render
from django.views import View

from .models import UserProfile
from django.conf import settings
from dtoken.views import make_token
from django.core.cache import cache
from .tasks import send_active_email_async

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

        #生成随机数
        code = "%s"%(random.randint(1000,9999))
        code_str = code + '_' + username
        active_code = base64.urlsafe_b64encode(code_str.encode())
        #存储随机数 - 3天有效期
        cache.set('email_active_%s'%(username), code, 60*60*24*3)
        verify_url = 'http://127.0.0.1:7000/dadashop/templates/active.html?code=%s'%(active_code.decode())
        print(verify_url)
        #发邮件
        #send_active_email(email, verify_url)
        send_active_email_async.delay(email, verify_url)

    except Exception as e:
        print('---active error---')
        print(e)
        pass

    return JsonResponse({'code':200,'username':username, 'data':{'token':token.decode()},'carts_count':0})


def send_active_email(email_address, v_url):
    #发激活邮件
    subject = '达达商城激活邮件'
    html_message = '''
    <p>尊敬的用户您好</p>
    <p>请点击此链接激活您的账户(3天内有效):</p>
    <p><a href='%s' target='_blank'>点击激活</a></p>
    '''%(v_url)
    send_mail(subject,'',from_email=settings.EMAIL_HOST_USER,recipient_list=[email_address],html_message=html_message,)


def active_view(request):

    if request.method != 'GET':
        return JsonResponse({'code':10103, 'error':'Please use GET'})
    #校验链接中的查询字符串
    #修改用户的 is_active 值

    code = request.GET.get('code')
    if not code:
        return JsonResponse({'code':10104, 'error':'no code'})

    verify_code = base64.urlsafe_b64decode(code.encode()).decode()
    #1238_username
    random_code, username = verify_code.split('_')
    old_code = cache.get('email_active_%s'%(username))
    if not old_code:
        return JsonResponse({'code':10105, 'error':'The link is invalid'})
    if old_code != random_code:
        return JsonResponse({'code':10106, 'error': 'The link is invalid!!'})

    user = UserProfile.objects.get(username=username)
    user.is_active = True
    user.save()
    #删除redis数据
    cache.delete('email_active_%s'%(username))
    return JsonResponse({'code':200, 'data':'ok'})

#FBV function base view
def address_view(request):
    # 增查 /v1/users/guoxiaonao/address
    # 改删 /v1/users/guoxiaonao/address/id

    if request.method == 'GET':
        #获取用户地址
        pass
    elif request.method == 'POST':
        #创建用户地址
        pass
    elif request.method == 'PUT':
        #更新用户地址
        pass
    elif request.method == 'DELETE':
        #删除地址
        pass

#CBV  class base view
class AddressView(View):
    #特点1 对于没有定义动作方法[def get]的请求，会直接返回 405的 response
    #特点2 所有该url的请求进入到视图类 统一先走 dispatch方法

    def dispatch(self, request, *args, **kwargs):
        #所有该url的请求进入到视图类 优先走该方法
        #集中的参数检查
        print('----dispatch do--')
        json_str = request.body
        if json_str:
            json_obj = json.loads(json_str)
            request.json_obj = json_obj

        return super().dispatch(request, *args, **kwargs)


    def get(self, request, username):
        print('---get--do---')
        return JsonResponse({'code':200})

    def post(self, request, username):
        data = request.json_obj
        #增加地址时, 用户的第一个地址 设置为默认地址
        pass









