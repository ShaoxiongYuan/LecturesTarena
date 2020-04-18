import base64
import hashlib
import json
import random
import time

from django.core.mail import send_mail
from django.http import JsonResponse
from django.shortcuts import render
from django.views import View

from .models import UserProfile, Address, WeiboProfile
from django.conf import settings
from dtoken.views import make_token
from django.core.cache import cache
from .tasks import send_active_email_async
from tools.logging_decorator import logging_check
from urllib.parse import urlencode
import requests
from django.db import transaction


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
    @logging_check
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
        user = request.myuser

        all_address = user.address_set.filter(is_active=True)
        res = []
        for add in all_address:
            add_dict = {}
            add_dict['id'] = add.id
            add_dict['address'] = add.address
            add_dict['receiver'] = add.receiver
            add_dict['receiver_mobile'] = add.receiver_mobile
            add_dict['tag'] = add.tag
            add_dict['is_default'] = add.is_default
            add_dict['postcode'] = add.postcode
            res.append(add_dict)
        return JsonResponse({'code':200, 'addresslist':res})



    def post(self, request, username):
        data = request.json_obj
        #增加地址时, 用户的第一个地址 设置为默认地址
        receiver = data['receiver']
        address = data['address']
        receiver_phone = data['receiver_phone']
        postcode = data['postcode']
        tag = data['tag']

        user = request.myuser

        is_default = False
        #old_address = user.address_set.all()
        old_address = Address.objects.filter(user_profile=user)
        if not old_address:
            is_default = True
        Address.objects.create(
            user_profile=user,
            receiver=receiver,
            address=address,
            is_default=is_default,
            receiver_mobile=receiver_phone,
            postcode=postcode,
            tag = tag,
        )
        return JsonResponse({'code':200, 'data':'新增地址成功!'})



def oauth_url_view(request):
    #获取微博授权地址
    params = {
        'client_id': settings.WEIBO_CLIENT_ID,
        'response_type': 'code',
        'redirect_uri': settings.WEIBO_REDIRECT_URI
    }

    weibo_url = 'https://api.weibo.com/oauth2/authorize?'
    oauth_url = weibo_url + urlencode(params)
    return JsonResponse({'code':200, 'oauth_url':oauth_url})


class OauthWeiboView(View):

    def get(self, request):
        #接收前端转发过来的 授权码
        code = request.GET.get('code')
        if not code:
            return JsonResponse({'code':10109, 'error':'no code'})
        token_url = 'https://api.weibo.com/oauth2/access_token'

        req_data = {
            'client_id': settings.WEIBO_CLIENT_ID,
            'client_secret': settings.WEIBO_CLIENT_SECRET,
            'grant_type': 'authorization_code',
            'redirect_uri': settings.WEIBO_REDIRECT_URI,
            'code': code
        }
        #向微博服务器发post请求，交换access_token
        response = requests.post(token_url, data=req_data)
        if response.status_code == 200:
            weibo_data = json.loads(response.text)
        else:
            print('weibo server error is %s'%(response.status_code))
            return JsonResponse({'code':10110, 'error':'Oauth error'})
        #如果返回数据中有 error字段，则证明交换失败
        if weibo_data.get('error'):
            #TODO 可按需打印出该异常信息
            return JsonResponse({'code':10111, 'error':'Oauth error!'})

        print(weibo_data)
        #{'access_token': '2.00rudyBCjfiLSD603d5ecdc40GBwIO', 'remind_in': '157679999', 'expires_in': 157679999, 'uid': '1861495121', 'isRealName': 'false'}

        weibo_uid = weibo_data['uid']
        access_token = weibo_data['access_token']
        #查询当前weibo用户是否是第一次 光临
        try:
            weibo_user = WeiboProfile.objects.get(wuid=weibo_uid)
        except Exception as e:
            #第一次来～ 插入数据
            WeiboProfile.objects.create(access_token=access_token,wuid=weibo_uid)
            return JsonResponse({'code':201, 'uid':weibo_uid})

        else:
            #之前来过，检查是否绑定过
            user = weibo_user.user_profile
            if user:
                #之前成功绑定过 UserProfile的用户
                username = user.username
                token = make_token(username)
                return JsonResponse({'code':200, 'username':username, 'token': token.decode()})
            else:
                #未绑定 - 触发绑定注册
                return JsonResponse({'code':201, 'uid':weibo_uid})


    def post(self, request):

        data = json.loads(request.body)
        uid = data['uid']
        username = data['username']
        password = data['password']
        phone = data['phone']
        email = data['email']

        m = hashlib.md5()
        m.update(password.encode())
        password_h = m.hexdigest()


        try:
            with transaction.atomic():
                # 创建UserProfile数据
                user = UserProfile.objects.create(username=username, password=password_h,email=email, phone=phone)
                # 绑定WeiboProfile的外键
                weibo_user = WeiboProfile.objects.get(wuid=uid)
                weibo_user.user_profile = user
                weibo_user.save()
        except Exception as e:
            print('--create error is %s'%(e))
            return JsonResponse({'code':10112, 'error':'Create user error'})

        token = make_token(username)
        return JsonResponse({'code':200, 'username':username, 'token': token.decode()})















